from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_as_class import InceptionResNetV2_progressive
from .base_model_as_class_3d import InceptionResNetV2_3D_Progressive
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .base_model_resnet import HighWayResnet2D, HighWayResnet2D_Progressive
from .layers import SkipUpsample3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import inception_resnet_block_3d, conv3d_bn, get_transformer_layer, HighwayMulti, EqualizedDense
from .multi_scale_task import MeanSTD, Sampling
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


def get_x2ct_model_ap_lat_v8(xray_shape, ct_series_shape,
                             block_size=16,
                             num_downsample=5,
                             base_act="leakyrelu",
                             last_act="tanh"):
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
                                             base_act=base_act,
                                             last_act=base_act,
                                             name_prefix="ap",
                                             num_downsample=num_downsample,
                                             use_attention=True)
    lat_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                              block_size=block_size,
                                              padding="same",
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
    attn_dim = C // attn_num_head

    ap_decoded = layers.Reshape((H * W, C))(ap_model_output)
    ap_decoded = AddPositionEmbs(input_shape=(H * W, C))(ap_decoded)

    lat_decoded = layers.Reshape((H * W, C))(lat_model_output)
    lat_decoded = AddPositionEmbs(input_shape=(H * W, C))(lat_decoded)

    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]

    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        ap_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=0)(ap_decoded)
        lat_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                         dropout=0)(lat_decoded)
    ap_decoded = layers.Reshape((ct_start_channel,
                                 ct_start_channel,
                                 ct_start_channel,
                                 H * W * C // (ct_start_channel ** 3)))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  H * W * C // (ct_start_channel ** 3)))(lat_decoded)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded])
    concat_decoded = conv3d_bn(concat_decoded, C, 3,
                               activation=base_act)

    for idx in range(5, 5 - num_downsample, -1):
        ap_skip_connect = ap_model.get_layer(f"ap_down_block_{idx}").output
        lat_skip_connect = lat_model.get_layer(f"lat_down_block_{idx}").output
        _, H, W, current_filter = ap_skip_connect.shape

        ap_skip_connect = SkipUpsample3D(current_filter,
                                         activation=base_act)(ap_skip_connect, H)
        lat_skip_connect = SkipUpsample3D(current_filter,
                                          activation=base_act)(lat_skip_connect, H)
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3,
                               activation=base_act)
        ap_decoded = layers.Concatenate()([ap_decoded, ap_skip_connect])
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3,
                                activation=base_act)
        lat_decoded = layers.Concatenate()([lat_decoded, lat_skip_connect])

        concat_decoded = layers.Concatenate()([concat_decoded,
                                               ap_decoded,
                                               lat_decoded])
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3,
                                   activation=base_act)

        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(concat_decoded)
        if idx > 5 - num_downsample - 1:
            ap_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(ap_decoded)
            lat_decoded = Decoder3D(current_filter,
                                    strides=(2, 2, 2),
                                    activation=base_act)(lat_decoded)
    output_tensor = conv3d_bn(concat_decoded, current_filter, 3,
                              activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    return Model([ap_model_input, lat_model_input], output_tensor)


def get_x2ct_model_ap_lat_v9(xray_shape, ct_series_shape,
                             block_size=16,
                             num_downsample=5,
                             base_act="leakyrelu",
                             last_act="tanh"):
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
                                             base_act=base_act,
                                             last_act=base_act,
                                             name_prefix="ap",
                                             num_downsample=num_downsample,
                                             use_attention=True)
    lat_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                              block_size=block_size,
                                              padding="same",
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
    attn_dim = C // attn_num_head

    ap_decoded = layers.Flatten()(ap_model_output)
    ap_decoded = EqualizedDense(H * W * C)(ap_decoded)
    ap_decoded = layers.Dropout(0.2)(ap_decoded)
    lat_decoded = layers.Flatten()(lat_model_output)
    lat_decoded = EqualizedDense(H * W * C)(lat_decoded)
    lat_decoded = layers.Dropout(0.2)(lat_decoded)

    ap_decoded = layers.Reshape((ct_start_channel,
                                 ct_start_channel,
                                 ct_start_channel,
                                 H * W * C // (ct_start_channel ** 3)))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  H * W * C // (ct_start_channel ** 3)))(lat_decoded)
    ap_decoded = conv3d_bn(ap_decoded, C, 3,
                           activation=base_act)
    lat_decoded = conv3d_bn(lat_decoded, C, 3,
                            activation=base_act)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded])
    concat_decoded = conv3d_bn(concat_decoded, C, 3,
                               activation=base_act)

    for idx in range(5, 5 - num_downsample, -1):
        ap_skip_connect = ap_model.get_layer(f"ap_down_block_{idx}").output
        lat_skip_connect = lat_model.get_layer(f"lat_down_block_{idx}").output
        _, H, W, current_filter = ap_skip_connect.shape

        ap_skip_connect = SkipUpsample3D(current_filter,
                                         activation=base_act)(ap_skip_connect, H)
        lat_skip_connect = SkipUpsample3D(current_filter,
                                          activation=base_act)(lat_skip_connect, H)
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3,
                               activation=base_act)
        ap_decoded = layers.Concatenate()([ap_decoded, ap_skip_connect])
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3,
                                activation=base_act)
        lat_decoded = layers.Concatenate()([lat_decoded, lat_skip_connect])

        concat_decoded = layers.Concatenate()([concat_decoded,
                                               ap_decoded,
                                               lat_decoded])
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3,
                                   activation=base_act)
        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(concat_decoded)
        if idx > 5 - num_downsample - 1:
            ap_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(ap_decoded)
            lat_decoded = Decoder3D(current_filter,
                                    strides=(2, 2, 2),
                                    activation=base_act)(lat_decoded)
    output_tensor = conv3d_bn(
        concat_decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    return Model([ap_model_input, lat_model_input], output_tensor)


def get_x2ct_model_ap_lat_v10(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              final_downsample=5,
                              base_act="relu",
                              decode_init_filter=64,
                              last_channel_activation="tanh"):

    model, skip_connect_name_list = HighWayResnet2D_Progressive(input_shape=(xray_shape[0], xray_shape[1], xray_shape[2]),
                                                                block_size=block_size,
                                                                num_downsample=num_downsample,
                                                                final_downsample=final_downsample,
                                                                padding="same",
                                                                base_act=base_act,
                                                                last_act=base_act,
                                                                name_prefix="x2ct"
                                                                )
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    model_input = model.input
    model_output = model.output
    skip_connection_outputs = [model.get_layer(layer_name).output
                               for layer_name in skip_connect_name_list]

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(model_output)
    ct_start_channel = xray_shape[0] // (2 ** num_downsample)
    ct_dim = ct_start_channel
    attn_num_head = 8
    attn_dim = C // attn_num_head

    decoded = layers.Reshape((H * W, C))(model_output)
    decoded = AddPositionEmbs(input_shape=(H * W, C))(decoded)

    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]

    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                     dropout=0)(decoded)

    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3)))(decoded)

    init_filter = decode_init_filter
    for index in range(num_downsample - 1, -1, -1):
        filter_index = num_downsample - index - 1
        current_filter = init_filter // (2 ** filter_index)
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)

        _, H, W, _, _ = backend.int_shape(decoded)
        skip_connect = skip_connection_outputs[index]
        skip_connect = SkipUpsample3D(current_filter,
                                      activation=base_act)(skip_connect, ct_dim)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)

        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
        ct_dim *= 2

    output_tensor = conv3d_bn(decoded, current_filter, 3, activation=base_act)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(model_input, output_tensor)


def get_x2ct_model_ap_lat_v11(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh"):

    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    model = InceptionResNetV2_progressive(target_shape=target_shape,
                                          block_size=block_size,
                                          padding="same",
                                          base_act=base_act,
                                          last_act=base_act,
                                          name_prefix="x2ct",
                                          num_downsample=num_downsample,
                                          use_attention=True)
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    model_input = model.input
    model_output = model.output
    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(model_output)
    ct_start_channel = xray_shape[0] // (2 ** num_downsample)
    ct_dim = ct_start_channel
    attn_num_head = 8
    attn_dim = C // attn_num_head

    decoded = layers.Reshape((H * W, C))(model_output)
    decoded = AddPositionEmbs(input_shape=(H * W, C))(decoded)

    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]

    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                     dropout=0)(decoded)
    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3)))(decoded)
    decoded = conv3d_bn(decoded, C, 3, activation=base_act)
    for idx in range(5, 5 - num_downsample, -1):
        skip_connect = model.get_layer(f"x2ct_down_block_{idx}").output
        _, H, W, current_filter = skip_connect.shape
        skip_connect = SkipUpsample3D(current_filter,
                                      activation=base_act)(skip_connect, H)
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)
        print(decoded.shape)
        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
    output_tensor = conv3d_bn(decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(model_input, output_tensor)


def get_x2ct_model_ap_lat_v12(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh",
                              latent_dim=2048):
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    model = InceptionResNetV2_progressive(target_shape=target_shape,
                                          block_size=block_size,
                                          padding="same",
                                          base_act=base_act,
                                          last_act=base_act,
                                          name_prefix="x2ct",
                                          num_downsample=num_downsample,
                                          use_attention=True)
    model_input = model.input
    model_output = model.output
    _, H, W, C = backend.int_shape(model_output)
    ct_start_channel = xray_shape[0] // (2 ** num_downsample)
    ct_dim = ct_start_channel

    mean_std_layer = MeanSTD(latent_dim=latent_dim, name="mean_var")
    sampling_layer = Sampling(name="sampling")
    decode_dense_layer = layers.Dense(H * W * C // 4,
                                      activation=tf.nn.relu6)
    # input.shape [B, 512, 512, 2]
    # output.shape: [B, 16, 16, 1536]
    decoded = conv2d_bn(model_output, C // 2, 3,
                        activation=base_act)
    decoded = conv2d_bn(decoded, C // 4, 3,
                        activation=base_act)
    z_mean, z_log_var = mean_std_layer(decoded)
    sampling_z = sampling_layer([z_mean, z_log_var])
    decoded = decode_dense_layer(sampling_z)

    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3) // 4))(decoded)
    decoded = conv3d_bn(decoded, C, 3, activation=base_act)

    for idx in range(5, 5 - num_downsample, -1):
        skip_connect = model.get_layer(f"x2ct_down_block_{idx}").output
        _, H, W, current_filter = skip_connect.shape
        skip_connect = SkipUpsample3D(current_filter,
                                      activation=base_act)(skip_connect, H)
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)

        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
    output_tensor = conv3d_bn(decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)

    return Model(model_input, output_tensor)


def get_x2ct_model_ap_lat_v13(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh",
                              latent_dim=2048):
    target_shape = (xray_shape[0] * (2 ** (5 - num_downsample)),
                    xray_shape[1] * (2 ** (5 - num_downsample)),
                    xray_shape[2])
    # element ratio: [256, 128, 96, 136, 65]
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    filter_list = [0.5, 1, 1.5, 1, 2]
    model = InceptionResNetV2_progressive(target_shape=target_shape,
                                          block_size=block_size,
                                          padding="same",
                                          base_act=base_act,
                                          last_act=base_act,
                                          name_prefix="x2ct",
                                          num_downsample=num_downsample,
                                          use_attention=True)
    model_input = model.input
    model_output = model.output
    _, H, W, C = backend.int_shape(model_output)
    ct_start_channel = xray_shape[0] // (2 ** num_downsample)
    ct_dim = ct_start_channel

    mean_std_layer = MeanSTD(latent_dim=latent_dim, name="mean_var")
    sampling_layer = Sampling(name="sampling")
    decode_dense_layer = layers.Dense(H * W * C // 4,
                                      activation=tf.nn.relu6)
    # input.shape [B, 512, 512, 2]
    # output.shape: [B, 16, 16, 1536]
    decoded = conv2d_bn(model_output, C // 2, 3,
                        activation=base_act)
    decoded = conv2d_bn(decoded, C // 4, 3,
                        activation=base_act)
    z_mean, z_log_var = mean_std_layer(decoded)
    sampling_z = sampling_layer([z_mean, z_log_var])
    decoded = decode_dense_layer(sampling_z)

    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3) // 4))(decoded)
    decoded = conv3d_bn(decoded, C, 3, activation=base_act)

    for idx in range(5, 5 - num_downsample, -1):
        skip_connect = model.get_layer(f"x2ct_down_block_{idx}").output
        _, H, W, current_filter = skip_connect.shape
        skip_connect = skip_recon_vae_block(skip_connect,
                                            int(latent_dim *
                                                filter_list[idx - 1] // 8),
                                            base_act=base_act,
                                            downscale=idx <= 1)
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)

        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
    output_tensor = conv3d_bn(decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(model_input, output_tensor)


def get_x2ct_model_ap_lat_v14(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh",
                              latent_dim=2048):
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
                                             base_act=base_act,
                                             last_act=base_act,
                                             name_prefix="ap",
                                             num_downsample=num_downsample,
                                             use_attention=True)
    lat_model = InceptionResNetV2_progressive(target_shape=(target_shape[0], target_shape[0], 1),
                                              block_size=block_size,
                                              padding="same",
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

    ap_mean_std_layer = MeanSTD(latent_dim=latent_dim, name="ap")
    lat_mean_std_layer = MeanSTD(latent_dim=latent_dim, name="lat")
    sampling_layer = Sampling(name="sampling")
    ap_decode_dense_layer = layers.Dense(H * W * C // 4,
                                         activation=tf.nn.relu6)
    lat_decode_dense_layer = layers.Dense(H * W * C // 4,
                                          activation=tf.nn.relu6)

    # input.shape [B, 512, 512, 2]
    # output.shape: [B, 16, 16, 1536]
    ap_decoded = conv2d_bn(ap_model_output, C // 2, 3,
                           activation=base_act)
    ap_decoded = conv2d_bn(ap_decoded, C // 4, 3,
                           activation=base_act)
    ap_z_mean, ap_z_log_var = ap_mean_std_layer(ap_decoded)

    lat_decoded = conv2d_bn(lat_model_output, C // 2, 3,
                            activation=base_act)
    lat_decoded = conv2d_bn(lat_decoded, C // 4, 3,
                            activation=base_act)
    lat_z_mean, lat_z_log_var = lat_mean_std_layer(lat_decoded)

    ap_sampling_z = sampling_layer([ap_z_mean, ap_z_log_var])
    lat_sampling_z = sampling_layer([lat_z_mean, lat_z_log_var])
    ap_decoded = ap_decode_dense_layer(ap_sampling_z)
    lat_decoded = lat_decode_dense_layer(lat_sampling_z)

    ap_decoded = layers.Reshape((ct_start_channel,
                                 ct_start_channel,
                                 ct_start_channel,
                                 H * W * C // (ct_start_channel ** 3) // 4))(ap_decoded)
    ap_decoded = conv3d_bn(ap_decoded, C, 3, activation=base_act)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  H * W * C // (ct_start_channel ** 3) // 4))(lat_decoded)
    lat_decoded = conv3d_bn(lat_decoded, C, 3, activation=base_act)

    concat_decoded = (ap_decoded + lat_decoded) / 2
    concat_decoded = conv3d_bn(concat_decoded, C, 3, activation=base_act)

    for idx in range(5, 5 - num_downsample, -1):
        ap_skip_connect = ap_model.get_layer(f"ap_down_block_{idx}").output
        lat_skip_connect = lat_model.get_layer(f"lat_down_block_{idx}").output
        _, H, W, current_filter = ap_skip_connect.shape

        ap_skip_connect = skip_recon_vae_block(ap_skip_connect,
                                               int(latent_dim *
                                                   filter_list[idx - 1] // 8),
                                               base_act=base_act,
                                               downscale=idx <= 1)
        lat_skip_connect = skip_recon_vae_block(lat_skip_connect,
                                                int(latent_dim *
                                                    filter_list[idx - 1] // 8),
                                                base_act=base_act,
                                                downscale=idx <= 1)
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3,
                               activation=base_act)
        ap_decoded = layers.Concatenate()([ap_decoded, ap_skip_connect])
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3,
                                activation=base_act)
        lat_decoded = layers.Concatenate()([lat_decoded, lat_skip_connect])

        concat_decoded = (ap_decoded + lat_decoded) / 2
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3,
                                   activation=base_act)

        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(concat_decoded)
        if idx > 5 - num_downsample - 1:
            ap_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2),
                                   activation=base_act)(ap_decoded)
            lat_decoded = Decoder3D(current_filter,
                                    strides=(2, 2, 2),
                                    activation=base_act)(lat_decoded)
    output_tensor = conv3d_bn(
        concat_decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1,
                              activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_model_input, lat_model_input], output_tensor)


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
                                                  base_act=base_act,
                                                  last_act=last_act,
                                                  name_prefix="x2ct",
                                                  num_downsample=num_downsample,
                                                  use_attention=True)

    return base_model


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
