from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_as_class import InceptionResNetV2_progressive, Conv2DBN
from .base_model_as_class_3d import EqualizedConv3D, InceptionResNetV2_3D_Progressive, Conv3DBN
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .layers import SkipUpsample3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import get_act_layer, conv3d_bn, get_transformer_layer, HighwayMulti, EqualizedDense
from .layers import PixelShuffleBlock3D, UpsampleBlock3D, TransposeBlock3D, SimpleOutputLayer2D, EqualizedConv
from .multi_scale_task import MeanSTD, Sampling
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


def get_x2ct_model_ap_lat_v8(xray_shape, ct_series_shape,
                             block_size=16,
                             pooling="average",
                             num_downsample=5,
                             base_act="leakyrelu",
                             last_act="tanh"):
    norm = "batch"
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    # element ratio: [256, 128, 96, 136, 65]
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    filter_list = [0.5, 1, 1.5, 1, 2]
    xray_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                               block_size=block_size,
                                               padding="same",
                                               pooling=pooling,
                                               norm=norm,
                                               base_act=base_act,
                                               last_act=base_act,
                                               name_prefix="xray",
                                               num_downsample=num_downsample,
                                               use_attention=True)

    xray_model_input = xray_model.input
    xray_model_output = xray_model.output
    _, H, W, C = backend.int_shape(xray_model_output)
    ct_start_channel = target_shape[0] // (2 ** 5)
    ct_dim = ct_start_channel

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(xray_model_output)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    xray_decoded = layers.Flatten()(xray_model_output)
    xray_decoded = EqualizedDense(H * W * C)(xray_decoded)
    # xray_decoded = layers.Dropout(0.2)(xray_decoded)

    xray_decoded = layers.Reshape((ct_start_channel,
                                   ct_start_channel,
                                   ct_start_channel,
                                   H * W * C // (ct_start_channel ** 3)))(xray_decoded)
    xray_decoded = Conv3DBN(C, 3,
                            norm=norm, activation=base_act)(xray_decoded)

    for idx in range(5, 5 - num_downsample, -1):
        xray_skip_connect = xray_model.get_layer(
            f"xray_down_block_{idx}").output
        _, H, W, current_filter = xray_skip_connect.shape

        xray_skip_connect = SkipUpsample3D(current_filter,
                                           norm=norm, activation=base_act)(xray_skip_connect, H)
        xray_decoded = Conv3DBN(current_filter, 3,
                                norm=norm, activation=base_act)(xray_decoded)
        xray_decoded = layers.Concatenate()([xray_decoded,
                                             xray_skip_connect])
        if idx > 5 - num_downsample - 1:
            xray_decoded = TransposeBlock3D(current_filter,
                                            strides=(2, 2, 2),
                                            norm=norm, activation=base_act)(xray_decoded)
        else:
            xray_decoded = Decoder3D(current_filter,
                                     strides=(2, 2, 2),
                                     norm=norm, activation=base_act)(xray_decoded)

    output_tensor = Conv3DBN(current_filter, 3,
                             norm=norm, activation=base_act)(xray_decoded)
    output_tensor = Conv3DBN(current_filter // 2, 1,
                             norm=norm, activation=None)(output_tensor)
    output_tensor = SimpleOutputLayer2D(last_channel_num=1,
                                        act=last_act)(output_tensor)
    return Model(xray_model_input, output_tensor)


def get_x2ct_model_ap_lat_v9(xray_shape, ct_series_shape,
                             block_size=16,
                             pooling="average",
                             num_downsample=5,
                             base_act="leakyrelu",
                             last_act="tanh"):
    norm = "batch"
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    # element ratio: [256, 128, 96, 136, 65]
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    filter_list = [0.5, 1, 1.5, 1, 2]
    ap_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                             block_size=block_size,
                                             padding="same",
                                             pooling=pooling,
                                             norm=norm,
                                             base_act=base_act,
                                             last_act=base_act,
                                             name_prefix="ap",
                                             num_downsample=num_downsample,
                                             use_attention=True)
    lat_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                              block_size=block_size,
                                              padding="same",
                                              pooling=pooling,
                                              norm=norm,
                                              base_act=base_act,
                                              last_act=base_act,
                                              name_prefix="lat",
                                              num_downsample=num_downsample,
                                              use_attention=True)

    ap_model_input = ap_model.input
    lat_model_input = lat_model.input
    ap_model_output = ap_model.output
    lat_model_output = lat_model.output
    _, H, W, C = backend.int_shape(ap_model_output)
    ct_start_channel = target_shape[0] // (2 ** 5)
    ct_dim = ct_start_channel

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(ap_model_output)
    attn_num_head = 8

    ap_decoded = layers.Flatten()(ap_model_output)
    print(ap_decoded.shape)
    ap_decoded = EqualizedDense(H * W * C)(ap_decoded)
    # ap_decoded = layers.Dropout(0.05)(ap_decoded)
    lat_decoded = layers.Flatten()(lat_model_output)
    lat_decoded = EqualizedDense(H * W * C)(lat_decoded)
    # lat_decoded = layers.Dropout(0.05)(lat_decoded)

    ap_decoded = layers.Reshape((ct_start_channel,
                                 ct_start_channel,
                                 ct_start_channel,
                                 H * W * C // (ct_start_channel ** 3)))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  H * W * C // (ct_start_channel ** 3)))(lat_decoded)
    ap_decoded = Conv3DBN(C, 3,
                          norm=norm, activation=base_act)(ap_decoded)
    lat_decoded = Conv3DBN(C, 3,
                           norm=norm, activation=base_act)(lat_decoded)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded])
    concat_decoded = Conv3DBN(C, 3,
                              norm=norm, activation=base_act)(concat_decoded)

    for idx in range(5, 5 - num_downsample, -1):
        ap_skip_connect = ap_model.get_layer(f"ap_down_block_{idx}").output
        lat_skip_connect = lat_model.get_layer(f"lat_down_block_{idx}").output
        _, H, W, current_filter = ap_skip_connect.shape

        ap_skip_connect = SkipUpsample3D(current_filter,
                                         norm=norm, activation=base_act)(ap_skip_connect, H)
        lat_skip_connect = SkipUpsample3D(current_filter,
                                          norm=norm, activation=base_act)(lat_skip_connect, H)
        ap_decoded = Conv3DBN(current_filter, 3,
                              norm=norm, activation=base_act)(ap_decoded)
        ap_decoded = layers.Concatenate()([ap_decoded,
                                           ap_skip_connect])
        lat_decoded = Conv3DBN(current_filter, 3,
                               norm=norm, activation=base_act)(lat_decoded)
        lat_decoded = layers.Concatenate()([lat_decoded,
                                            lat_skip_connect])

        concat_decoded = layers.Concatenate()([concat_decoded,
                                               ap_decoded,
                                               lat_decoded])
        concat_decoded = Conv3DBN(current_filter, 3,
                                  norm=norm, activation=base_act)(concat_decoded)
        if idx > 5 - num_downsample - 1:
            concat_decoded = TransposeBlock3D(current_filter,
                                              strides=(2, 2, 2),
                                              norm=norm, activation=base_act)(concat_decoded)
            ap_decoded = TransposeBlock3D(current_filter,
                                          strides=(2, 2, 2),
                                          norm=norm, activation=base_act)(ap_decoded)
            lat_decoded = TransposeBlock3D(current_filter,
                                           strides=(2, 2, 2),
                                           norm=norm, activation=base_act)(lat_decoded)
        else:
            concat_decoded = Decoder3D(current_filter,
                                       strides=(2, 2, 2),
                                       norm=norm, activation=base_act)(concat_decoded)

    output_tensor = Conv3DBN(current_filter, 3,
                             norm=norm, activation=base_act)(concat_decoded)
    output_tensor = Conv3DBN(current_filter // 2, 1,
                             norm=norm, activation=None)(output_tensor)
    output_tensor = SimpleOutputLayer2D(last_channel_num=1,
                                        act=last_act)(output_tensor)
    return Model([ap_model_input, lat_model_input], output_tensor)


def get_x2ct_model_ap_lat_v10(xray_shape, ct_series_shape,
                              block_size=16,
                              pooling="average",
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh"):
    norm = "batch"
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    # element ratio: [256, 128, 96, 136, 65]
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    filter_list = [0.5, 1, 1.5, 1, 2]
    ap_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                             block_size=block_size,
                                             padding="same",
                                             pooling=pooling,
                                             norm=norm,
                                             base_act=base_act,
                                             last_act=base_act,
                                             name_prefix="ap",
                                             num_downsample=num_downsample,
                                             use_attention=True)
    lat_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                              block_size=block_size,
                                              padding="same",
                                              pooling=pooling,
                                              norm=norm,
                                              base_act=base_act,
                                              last_act=base_act,
                                              name_prefix="lat",
                                              num_downsample=num_downsample,
                                              use_attention=True)

    ap_model_input = ap_model.input
    lat_model_input = lat_model.input
    ap_model_output = ap_model.output
    lat_model_output = lat_model.output
    _, H, W, C = backend.int_shape(ap_model_output)
    ct_start_channel = target_shape[0] // (2 ** 5)
    ct_dim = ct_start_channel

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(ap_model_output)
    down_channel = C // 3
    attn_num_head = 8

    ap_decoded = layers.Flatten()(ap_model_output)
    ap_decoded = EqualizedDense(H * W * down_channel)(ap_decoded)
    # ap_decoded = layers.Dropout(0.05)(ap_decoded)
    lat_decoded = layers.Flatten()(lat_model_output)
    lat_decoded = EqualizedDense(H * W * down_channel)(lat_decoded)
    # lat_decoded = layers.Dropout(0.05)(lat_decoded)

    ap_decoded = layers.Reshape((ct_start_channel,
                                 ct_start_channel,
                                 ct_start_channel,
                                 H * W * down_channel // (ct_start_channel ** 3)))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  H * W * down_channel // (ct_start_channel ** 3)))(lat_decoded)
    ap_decoded = Conv3DBN(C, 3,
                          norm=norm, activation=base_act)(ap_decoded)
    lat_decoded = Conv3DBN(C, 3,
                           norm=norm, activation=base_act)(lat_decoded)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded])
    concat_decoded = Conv3DBN(C, 3,
                              norm=norm, activation=base_act)(concat_decoded)

    # for idx in range(5 - num_downsample):
    #     current_filter = C
    #     ap_decoded = Conv3DBN(current_filter, 3,
    #                           norm=norm, activation=base_act)(ap_decoded)
    #     lat_decoded = Conv3DBN(current_filter, 3,
    #                            norm=norm, activation=base_act)(lat_decoded)
    #     concat_decoded = Conv3DBN(current_filter, 3,
    #                               norm=norm, activation=base_act)(concat_decoded)
    #     concat_decoded = layers.Concatenate()([concat_decoded,
    #                                            ap_decoded,
    #                                            lat_decoded])
    #     concat_decoded = Conv3DBN(current_filter, 3,
    #                               norm=norm, activation=base_act)(concat_decoded)
    #     concat_decoded = TransposeBlock3D(current_filter,
    #                                       strides=(2, 2, 2),
    #                                       norm=norm, activation=base_act)(concat_decoded)
    #     ap_decoded = TransposeBlock3D(current_filter,
    #                                   strides=(2, 2, 2),
    #                                   norm=norm, activation=base_act)(ap_decoded)
    #     lat_decoded = TransposeBlock3D(current_filter,
    #                                    strides=(2, 2, 2),
    #                                    norm=norm, activation=base_act)(lat_decoded)

    for idx in range(5, 5 - num_downsample, -1):
        ap_skip_connect = ap_model.get_layer(f"ap_block_{idx}").output
        lat_skip_connect = lat_model.get_layer(f"lat_block_{idx}").output
        _, H, W, current_filter = ap_skip_connect.shape

        ap_decoded = Conv3DBN(current_filter, 3,
                              norm=norm, activation=base_act)(ap_decoded)

        lat_decoded = Conv3DBN(current_filter, 3,
                               norm=norm, activation=base_act)(lat_decoded)

        concat_decoded = layers.Concatenate()([concat_decoded,
                                               ap_decoded,
                                               lat_decoded])
        concat_decoded = Conv3DBN(current_filter, 3,
                                  norm=norm, activation=base_act)(concat_decoded)

        ap_skip_connect = SkipUpsample3D(current_filter,
                                         norm=norm, activation=base_act)(ap_skip_connect, H)
        lat_skip_connect = SkipUpsample3D(current_filter,
                                          norm=norm, activation=base_act)(lat_skip_connect, H)

        if idx % 2 == 0:
            decode_block = PixelShuffleBlock3D
        else:
            decode_block = UpsampleBlock3D

        if idx > 5 - num_downsample - 1:
            concat_decoded = decode_block(current_filter,
                                          strides=(2, 2, 2),
                                          norm=norm, activation=base_act)(concat_decoded)
            ap_decoded = decode_block(current_filter,
                                      strides=(2, 2, 2),
                                      norm=norm, activation=base_act)(ap_decoded)
            lat_decoded = decode_block(current_filter,
                                       strides=(2, 2, 2),
                                       norm=norm, activation=base_act)(lat_decoded)
            ap_decoded = layers.Concatenate()([ap_decoded,
                                               ap_skip_connect])
            lat_decoded = layers.Concatenate()([lat_decoded,
                                                lat_skip_connect])

        else:
            concat_decoded = decode_block(current_filter,
                                          strides=(2, 2, 2),
                                          norm=norm, activation=base_act)(concat_decoded)
            concat_decoded = layers.Concatenate()([concat_decoded,
                                                   ap_skip_connect,
                                                   lat_skip_connect])

    output_tensor = Conv3DBN(current_filter, 3,
                             norm=norm, activation=base_act)(concat_decoded)
    output_tensor = Conv3DBN(current_filter // 2, 1,
                             norm=norm, activation=None)(output_tensor)
    output_tensor = SimpleOutputLayer2D(last_channel_num=1,
                                        act=last_act)(output_tensor)
    return Model([ap_model_input, lat_model_input], output_tensor)


def get_x2ct_model_ap_lat_v11(xray_shape, ct_series_shape,
                              block_size=16,
                              start_channel_ratio=1,
                              decoder_filter_ratio=1,
                              pooling="average",
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh"):
    norm = "batch"
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    base_model = InceptionResNetV2_progressive(target_shape=target_shape,
                                               block_size=block_size,
                                               padding="same",
                                               pooling=pooling,
                                               norm=norm,
                                               base_act=base_act,
                                               last_act=base_act,
                                               name_prefix="xray",
                                               num_downsample=num_downsample,
                                               use_attention=True)

    base_model_input = base_model.input
    base_model_output = base_model.output
    _, H, W, C = backend.int_shape(base_model_output)
    ct_start_channel = target_shape[0] // (2 ** 5)

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(base_model_output)
    down_channel = int(round(C // 3 * start_channel_ratio))

    decoded = EqualizedConv(H * W * down_channel)(base_model_output)
    decoded = layers.Flatten()(base_model_output)
    decoded = EqualizedDense(H * W * down_channel)(decoded)

    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * down_channel // (ct_start_channel ** 3)))(decoded)
    decoded = Conv3DBN(C, 3,
                       norm=norm, activation=base_act)(decoded)

    for idx in range(5, 5 - num_downsample, -1):
        skip_connect = base_model.get_layer(f"xray_block_{idx}").output
        _, H, W, current_filter = skip_connect.shape
        current_filter = int(round(current_filter * decoder_filter_ratio))
        decoded = Conv3DBN(current_filter, 3,
                           norm=norm, activation=base_act)(decoded)

        skip_connect = SkipUpsample3D(current_filter,
                                      norm=norm, activation=base_act)(skip_connect, H)
        if idx % 2 == 0:
            decoded = UpsampleBlock3D(current_filter,
                                      strides=(2, 2, 2),
                                      norm=norm, activation=base_act)(decoded)

        else:
            decoded = PixelShuffleBlock3D(current_filter,
                                          strides=(2, 2, 2),
                                          norm=norm, activation=base_act)(decoded)
        if idx == 5 - num_downsample + 1:
            pass
        else:
            decoded = layers.Concatenate()([decoded,
                                            skip_connect])
    output_tensor = Conv3DBN(current_filter, 3,
                             norm=norm, activation=None)(decoded)
    output_tensor = SimpleOutputLayer2D(last_channel_num=1,
                                        act=last_act)(output_tensor)
    return Model(base_model_input, output_tensor)


def skip_recon_vae_block(input_tensor, latent_dim,
                         base_act, downscale=False, name=None):
    _, H, W, C = backend.int_shape(input_tensor)

    mean_std_layer = MeanSTD(latent_dim=latent_dim, name=None)
    sampling_layer = Sampling(name=None)
    decode_dense_layer = layers.Dense(H * W * H * C // 4,
                                      activation=tf.nn.relu6)
    reshape_layer = layers.Reshape((H, W, H,
                                    C // 4))
    model_input = layers.Input(input_tensor.shape[1:])
    if downscale:
        scale = conv2d_bn(model_input, C // 4, 3,
                          activation=base_act)
    else:
        scale = model_input
    z_mean, z_log_var = mean_std_layer(scale)
    sampling_z = sampling_layer([z_mean, z_log_var])
    decoded = decode_dense_layer(sampling_z)
    decoded = reshape_layer(decoded)
    decoded = conv2d_bn(decoded, C, 3,
                        activation=base_act)
    return Model(model_input, decoded, name=name)(input_tensor)


def get_inception_resnet_v2_disc(input_shape,
                                 block_size=16,
                                 num_downsample=5,
                                 padding="valid",
                                 norm="instance",
                                 validity_act=None,
                                 base_act="leakyrelu",
                                 last_act="sigmoid",
                                 ):
    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2] * (2 ** (5 - num_downsample)),
                    input_shape[3])
    base_model = InceptionResNetV2_3D_Progressive(target_shape=target_shape,
                                                  block_size=block_size,
                                                  padding=padding,
                                                  norm=norm,
                                                  base_act=base_act,
                                                  last_act=last_act,
                                                  name_prefix="validity",
                                                  num_downsample=num_downsample,
                                                  use_attention=True)

    base_input = base_model.input
    base_output = base_model.output

    validity_pred = EqualizedConv3D(1, kernel=3)(base_output)
    validity_pred = get_act_layer(validity_act)(validity_pred)
    model = Model(base_input, validity_pred)

    return model


def get_inception_resnet_v2_disc_2d(input_shape,
                                    block_size=16,
                                    num_downsample=5,
                                    norm="instance",
                                    padding="valid",
                                    validity_act=None,
                                    base_act="leakyrelu",
                                    last_act="sigmoid",
                                    ):
    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2])
    base_model = InceptionResNetV2_progressive(target_shape=target_shape,
                                               block_size=block_size,
                                               padding=padding,
                                               norm=norm,
                                               base_act=base_act,
                                               last_act=last_act,
                                               name_prefix="validity",
                                               num_downsample=num_downsample,
                                               use_attention=True)

    base_input = base_model.input
    base_output = base_model.output

    validity_pred = EqualizedConv(1, kernel=3)(base_output)
    validity_pred = get_act_layer(validity_act)(validity_pred)
    model = Model(base_input, validity_pred)

    return model


def get_ct2x_model(ct_series_shape,
                   block_size=16,
                   decode_init_filter=768,
                   skip_connect=True,
                   base_act="leakyrelu",
                   last_channel_num=1,
                   last_channel_activation="sigmoid"
                   ):

    ct_start_channel = ct_series_shape[0] // 32

    base_model = InceptionResNetV2_3D(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(*ct_series_shape, 1),
        block_size=block_size,
        classes=None,
        padding="same",
        base_act=base_act,
        last_act=base_act,
        pooling=None,
        classifier_activation=None,
        attention_module=None
    )
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]
    init_filter = decode_init_filter
    decoded = base_output

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, Z, C = backend.int_shape(decoded)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    decoded = layers.Reshape((H * W * Z, C))(decoded)
    decoded = AddPositionEmbs(input_shape=(H * W * Z, C))(decoded)

    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]

    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                     dropout=0)(decoded)

    current_filter = block_size * 6 * ct_start_channel ** 3
    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              current_filter))(decoded)

    for index in range(0, 5):
        if skip_connect:
            # B H W Z C
            skip_connect_output = skip_connection_outputs[4 - index]
            skip_connect_ap_output = backend.mean(skip_connect_output, axis=1)
            skip_connect_ap_output = conv2d_bn(
                skip_connect_ap_output, current_filter // 2, 3)
            skip_connect_lat_output = backend.mean(skip_connect_output, axis=2)
            skip_connect_lat_output = conv2d_bn(
                skip_connect_lat_output, current_filter // 2, 3)
            skip_connect_output = layers.Concatenate(axis=-1)([skip_connect_ap_output,
                                                               skip_connect_lat_output])
            decoded = HighwayMulti(dim=current_filter, mode='2d')(
                decoded, skip_connect_output)
        current_filter = init_filter // (2 ** index)
        decoded = conv2d_bn(decoded, current_filter, 3)
        decoded = Decoder2D(current_filter,
                            kernel_size=2)(decoded)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(decoded)
    return Model(base_input, output_tensor)
