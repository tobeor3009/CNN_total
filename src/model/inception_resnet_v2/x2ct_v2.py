from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .base_model_resnet import HighWayResnet2D
from .layers import SkipUpsample3D, HighwayResnetDecoder3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import inception_resnet_block_3d, conv3d_bn, get_transformer_layer, HighwayMulti
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


def get_x2ct_model_ap_lat_v6(xray_shape, ct_series_shape,
                             block_size=16,
                             base_act="relu",
                             decode_init_filter=64,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
        base_act=base_act,
        last_act=base_act,
        classifier_activation=None,
        attention_module=None,
        name_prefix="ap"
    )
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    ap_input = ap_model.input
    ap_output = ap_model.output
    ap_skip_connection_outputs = [ap_model.get_layer(f"ap_{layer_name}").output
                                  for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    lat_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
        base_act=base_act,
        last_act=base_act,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    ct_start_channel = xray_shape[0] // 32
    ct_dim = ct_start_channel

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(ap_output)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    ap_decoded = layers.Reshape((H * W, C))(ap_output)
    ap_decoded = AddPositionEmbs(input_shape=(H * W, C))(ap_decoded)

    lat_decoded = layers.Reshape((H * W, C))(lat_output)
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
                                 block_size * 6 * ct_start_channel))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  block_size * 6 * ct_start_channel))(lat_decoded)
    concat_decoded = (ap_decoded + lat_decoded) / math.sqrt(2)

    init_filter = decode_init_filter
    for index in range(5):
        current_filter = init_filter // (2 ** index)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        _, H, W, _, _ = backend.int_shape(ap_decoded)
        ap_skip_connect = ap_skip_connection_outputs[4 - index]
        lat_skip_connect = lat_skip_connection_outputs[4 - index]
        ap_skip_connect = SkipUpsample3D(
            current_filter)(ap_skip_connect, ct_dim)
        lat_skip_connect = SkipUpsample3D(
            current_filter)(lat_skip_connect, ct_dim)

        ap_decoded = (ap_decoded + ap_skip_connect) / math.sqrt(2)
        lat_decoded = (lat_decoded + lat_skip_connect) / math.sqrt(2)
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)

        concat_decoded = (ap_decoded + lat_decoded +
                          concat_decoded) / math.sqrt(3)

        ap_decoded = Decoder3D(current_filter,
                               strides=(2, 2, 2))(ap_decoded)
        lat_decoded = Decoder3D(current_filter,
                                strides=(2, 2, 2))(lat_decoded)
        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2))(concat_decoded)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v7(xray_shape, ct_series_shape,
                             block_size=16,
                             base_act="relu",
                             decode_init_filter=64,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
        base_act=base_act,
        last_act=base_act,
        classifier_activation=None,
        name_prefix="ap"
    )
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    ap_input = ap_model.input
    ap_output = ap_model.output
    ap_skip_connection_outputs = [ap_model.get_layer(f"ap_{layer_name}").output
                                  for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    lat_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
        base_act=base_act,
        last_act=base_act,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    ct_start_channel = xray_shape[0] // 32
    ct_dim = ct_start_channel

    init_filter = decode_init_filter
    ap_decoded = SkipUpsample3D(init_filter)(ap_output, ct_dim)
    ap_decoded = conv3d_bn(ap_decoded, decode_init_filter, 3)

    lat_decoded = SkipUpsample3D(init_filter)(lat_output, ct_dim)
    lat_decoded = conv3d_bn(lat_decoded, decode_init_filter, 3)

    concat_decoded = (ap_decoded + lat_decoded) / math.sqrt(2)

    init_filter = decode_init_filter
    for index in range(5):
        current_filter = init_filter // (2 ** index)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        ap_skip_connect = ap_skip_connection_outputs[4 - index]
        lat_skip_connect = lat_skip_connection_outputs[4 - index]
        ap_skip_connect = SkipUpsample3D(
            current_filter)(ap_skip_connect, ct_dim)
        lat_skip_connect = SkipUpsample3D(
            current_filter)(lat_skip_connect, ct_dim)

        ap_decoded = HighwayMulti(dim=current_filter, mode='3d')(
            ap_decoded, ap_skip_connect)
        lat_decoded = HighwayMulti(dim=current_filter, mode='3d')(
            lat_decoded, lat_skip_connect)
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)

        concat_decoded = (ap_decoded + lat_decoded +
                          concat_decoded) / math.sqrt(3)

        ap_decoded = Decoder3D(current_filter,
                               strides=(2, 2, 2))(ap_decoded)
        lat_decoded = Decoder3D(current_filter,
                                strides=(2, 2, 2))(lat_decoded)
        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2))(concat_decoded)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v8(xray_shape, ct_series_shape,
                             block_size=16,
                             base_act="relu",
                             decode_init_filter=64,
                             last_channel_activation="tanh"):

    num_downsample = 5

    ap_model, ap_skip_connect_name_list = HighWayResnet2D(input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
                                                          block_size=16,
                                                          num_downsample=num_downsample,
                                                          padding="same",
                                                          base_act=base_act,
                                                          last_act=base_act,
                                                          name_prefix="ap"
                                                          )
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    ap_input = ap_model.input
    ap_output = ap_model.output
    ap_skip_connection_outputs = [ap_model.get_layer(layer_name).output
                                  for layer_name in ap_skip_connect_name_list]
    lat_model, lat_skip_connect_name_list = HighWayResnet2D(input_shape=(xray_shape[0], xray_shape[1], xray_shape[2] // 2),
                                                            block_size=16,
                                                            num_downsample=num_downsample,
                                                            padding="same",
                                                            base_act=base_act,
                                                            last_act=base_act,
                                                            name_prefix="lat"
                                                            )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(layer_name).output
                                   for layer_name in lat_skip_connect_name_list]

    ct_start_channel = xray_shape[0] // (2 ** num_downsample)
    ct_dim = ct_start_channel

    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(ap_output)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    ap_decoded = layers.Reshape((H * W, C))(ap_output)
    ap_decoded = AddPositionEmbs(input_shape=(H * W, C))(ap_decoded)

    lat_decoded = layers.Reshape((H * W, C))(lat_output)
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
                                 block_size * ct_start_channel * 2))(ap_decoded)
    lat_decoded = layers.Reshape((ct_start_channel,
                                  ct_start_channel,
                                  ct_start_channel,
                                  block_size * ct_start_channel * 2))(lat_decoded)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded])

    init_filter = decode_init_filter

    ap_decoded = conv3d_bn(ap_decoded, init_filter, 3)
    lat_decoded = conv3d_bn(lat_decoded, init_filter, 3)
    concat_decoded = conv3d_bn(concat_decoded, init_filter, 3)
    ap_decoded = conv3d_bn(ap_decoded, init_filter, 3)
    lat_decoded = conv3d_bn(lat_decoded, init_filter, 3)

    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded,
                                           concat_decoded])
    ap_decoded = Decoder3D(init_filter,
                           strides=(2, 2, 2))(ap_decoded)
    lat_decoded = Decoder3D(init_filter,
                            strides=(2, 2, 2))(lat_decoded)
    concat_decoded = Decoder3D(init_filter,
                               strides=(2, 2, 2))(concat_decoded)
    concat_decoded = conv3d_bn(concat_decoded, init_filter, 3)
    ct_dim *= 2
    for index in range(1, num_downsample):
        current_filter = init_filter // (2 ** index)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        _, H, W, _, _ = backend.int_shape(ap_decoded)
        ap_skip_connect = ap_skip_connection_outputs[-index]
        lat_skip_connect = lat_skip_connection_outputs[-index]
        ap_skip_connect = SkipUpsample3D(
            current_filter)(ap_skip_connect, ct_dim)
        lat_skip_connect = SkipUpsample3D(
            current_filter)(lat_skip_connect, ct_dim)

        ap_decoded = layers.Concatenate()([ap_decoded, ap_skip_connect])
        lat_decoded = layers.Concatenate()([lat_decoded, lat_skip_connect])
        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)

        concat_decoded = layers.Concatenate()([ap_decoded,
                                               lat_decoded,
                                               concat_decoded])

        ap_decoded = Decoder3D(current_filter,
                               strides=(2, 2, 2))(ap_decoded)
        lat_decoded = Decoder3D(current_filter,
                                strides=(2, 2, 2))(lat_decoded)
        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2))(concat_decoded)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2
    ap_skip_connect = ap_skip_connection_outputs[0]
    lat_skip_connect = lat_skip_connection_outputs[0]
    ap_skip_connect = SkipUpsample3D(
        current_filter)(ap_skip_connect, ct_dim)
    lat_skip_connect = SkipUpsample3D(
        current_filter)(lat_skip_connect, ct_dim)
    concat_decoded = layers.Concatenate()([ap_decoded,
                                           lat_decoded,
                                           concat_decoded])
    output_tensor = conv3d_bn(concat_decoded, current_filter, 3)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


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
