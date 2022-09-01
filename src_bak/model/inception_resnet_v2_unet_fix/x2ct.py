from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .layers import SkipUpsample3D, HighwayResnetDecoder3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import inception_resnet_block_3d, conv3d_bn, get_transformer_layer, HighwayMulti
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


def get_x2ct_model(xray_shape, ct_series_shape,
                   block_size=16,
                   base_act="relu",
                   desne_middle_filter=256,
                   dense_middle_size=3,
                   decode_init_filter=64,
                   include_context=False,
                   last_channel_activation="tanh"):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=xray_shape,
        block_size=block_size,
        classes=None,
        padding="same",
        base_act=base_act,
        last_act=base_act,
        pooling=None,
        classifier_activation=None
    )
    # base_input.shape [B, 512, 512, 2]
    # base_output.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    ct_start_channel = 16
    x = conv2d_bn(base_output, desne_middle_filter, 3)
    x = layers.Flatten()(x)
    x = layers.Dense(ct_start_channel ** 3 * dense_middle_size)(x)
    x = layers.Reshape(
        (ct_start_channel, ct_start_channel, ct_start_channel, dense_middle_size))(x)
    x = conv3d_bn(x, decode_init_filter, 3)

    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    init_filter = decode_init_filter
    ct_dim = ct_start_channel
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)
        x = conv3d_bn(x, current_filter, 3)
        skip_connect = skip_connection_outputs[4 - index]
        skip_connect = SkipUpsample3D(current_filter,
                                      include_context=include_context)(skip_connect, ct_dim)
        x = layers.Concatenate(axis=-1)([x, skip_connect])
        x = Decoder3D(current_filter,
                      strides=(2, 2, 2))(x)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(x)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(base_input, output_tensor)


def get_x2ct_model_ap_lat(xray_shape, ct_series_shape,
                          decode_init_filter=64,
                          include_context=False,
                          last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
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
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    concat_output = layers.Concatenate(axis=-1)([ap_output, lat_output])

    ct_start_channel = 16
    # x.shape: [B, 16, 16, 16, 1536]
    ap_decoded = SkipUpsample3D(filters=1536,
                                include_context=include_context)(ap_output, ct_start_channel)
    for block_idx in range(1, 6):
        ap_decoded = inception_resnet_block_3d(ap_decoded, scale=0.2,
                                               block_type='block8_3d',
                                               block_idx=f"ap_{block_idx}")
    lat_decoded = SkipUpsample3D(filters=1536,
                                 include_context=include_context)(lat_output, ct_start_channel)
    for block_idx in range(1, 6):
        lat_decoded = inception_resnet_block_3d(lat_decoded, scale=0.2,
                                                block_type='block8_3d',
                                                block_idx=f"lat_{block_idx}")
    concat_decoded = SkipUpsample3D(filters=1536,
                                    include_context=include_context)(concat_output, ct_start_channel)
    for block_idx in range(1, 6):
        concat_decoded = inception_resnet_block_3d(concat_decoded, scale=0.2,
                                                   block_type='block8_3d',
                                                   block_idx=f"concat_{block_idx}")

    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    init_filter = decode_init_filter
    ct_dim = ct_start_channel
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        ap_skip_connect = ap_skip_connection_outputs[4 - index]
        ap_skip_connect = SkipUpsample3D(current_filter,
                                         include_context=include_context)(ap_skip_connect, ct_dim)
        lat_skip_connect = lat_skip_connection_outputs[4 - index]
        lat_skip_connect = SkipUpsample3D(current_filter,
                                          include_context=include_context)(lat_skip_connect, ct_dim)
        ap_decoded = layers.Concatenate(axis=-1)([ap_decoded, ap_skip_connect])
        ap_decoded = HighwayResnetDecoder3D(current_filter,
                                            strides=(2, 2, 2))(ap_decoded)
        lat_decoded = layers.Concatenate(
            axis=-1)([lat_decoded, lat_skip_connect])
        lat_decoded = HighwayResnetDecoder3D(current_filter,
                                             strides=(2, 2, 2))(lat_decoded)

        concat_decoded = HighwayResnetDecoder3D(current_filter,
                                                strides=(2, 2, 2))(concat_decoded)
        concat_decoded = layers.Concatenate(
            axis=-1)([ap_decoded, lat_decoded, concat_decoded])
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v2(xray_shape, ct_series_shape,
                             decode_init_filter=64,
                             include_context=False,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
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
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    if include_context:
        _, H, W, C = backend.int_shape(ap_output)
        attn_num_head = 8
        attn_dim = C // attn_num_head
        ap_decoded = tf.nn.depth_to_space(ap_output, block_size=2)
        ap_decoded = layers.Reshape((H * W, C))(ap_decoded)
        ap_decoded = get_transformer_layer(ap_decoded, num_layer=6, heads=8, dim_head=96,
                                           hidden_dim=96 * 8 * 2, dropout=0.)

        lat_decoded = tf.nn.depth_to_space(lat_output, block_size=2)
        lat_decoded = layers.Reshape((H * W, C))(lat_decoded)
        lat_decoded = get_transformer_layer(lat_decoded, num_layer=6, heads=8, dim_head=96,
                                            hidden_dim=96 * 8 * 2, dropout=0.)

        ap_decoded = layers.Reshape((H * 2, H * 2, C // 4))(ap_decoded)
        ap_decoded = tf.nn.space_to_depth(ap_decoded, block_size=2)
        lat_decoded = layers.Reshape((H * 2, H * 2, C // 4))(lat_decoded)
        lat_decoded = tf.nn.space_to_depth(lat_decoded, block_size=2)

        ap_output = ap_decoded
        lat_output = lat_decoded

    concat_output = layers.Concatenate(axis=-1)([ap_output, lat_output])
    ct_start_channel = 16
    # x.shape: [B, 16, 16, 16, 1536]
    ap_decoded = SkipUpsample3D(filters=1536)(ap_output, ct_start_channel)
    for block_idx in range(1, 6):
        ap_decoded = inception_resnet_block_3d(ap_decoded, scale=0.2,
                                               block_type='block8_3d',
                                               block_idx=f"ap_{block_idx}")
    lat_decoded = SkipUpsample3D(filters=1536)(lat_output, ct_start_channel)
    for block_idx in range(1, 6):
        lat_decoded = inception_resnet_block_3d(lat_decoded, scale=0.2,
                                                block_type='block8_3d',
                                                block_idx=f"lat_{block_idx}")
    concat_decoded = SkipUpsample3D(filters=1536)(
        concat_output, ct_start_channel)

    for block_idx in range(1, 6):
        concat_decoded = inception_resnet_block_3d(concat_decoded, scale=0.2,
                                                   block_type='block8_3d',
                                                   block_idx=f"concat_{block_idx}")

    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    init_filter = decode_init_filter
    ct_dim = ct_start_channel
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        ap_skip_connect = ap_skip_connection_outputs[4 - index]
        ap_skip_connect = SkipUpsample3D(
            current_filter)(ap_skip_connect, ct_dim)
        lat_skip_connect = lat_skip_connection_outputs[4 - index]
        lat_skip_connect = SkipUpsample3D(
            current_filter)(lat_skip_connect, ct_dim)
        ap_decoded = layers.Concatenate(axis=-1)([ap_decoded, ap_skip_connect])
        ap_decoded = HighwayResnetDecoder3D(current_filter,
                                            strides=(2, 2, 2))(ap_decoded)
        lat_decoded = layers.Concatenate(
            axis=-1)([lat_decoded, lat_skip_connect])
        lat_decoded = HighwayResnetDecoder3D(current_filter,
                                             strides=(2, 2, 2))(lat_decoded)

        concat_decoded = HighwayResnetDecoder3D(current_filter,
                                                strides=(2, 2, 2))(concat_decoded)
        concat_decoded = layers.Concatenate(
            axis=-1)([ap_decoded, lat_decoded, concat_decoded])
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v3(xray_shape, ct_series_shape,
                             decode_init_filter=64,
                             include_context=False,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="ap"
    )
    # ap_output.sha
    # ap_output.shape: [B, 16, 16, 1536]
    ap_input = ap_model.input
    ap_output = ap_model.output
    ap_skip_connection_outputs = [ap_model.get_layer(f"ap_{layer_name}").output
                                  for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    lat_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    if include_context:
        # lat_output.shape: [B, 16, 16, 1536]
        _, H, W, C = backend.int_shape(ap_output)
        attn_num_head = 8
        attn_dim = C // attn_num_head
        # concat_decoded.shape: [B, 16, 16, 3072]
        concat_decoded = layers.Concatenate(axis=-1)([ap_output, lat_output])
        # concat_decoded.shape: [B, 32, 32, 768]
        concat_decoded = tf.nn.depth_to_space(concat_decoded, block_size=2)
        concat_decoded = layers.Reshape((H * W * 4, C // 2))(concat_decoded)

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
            concat_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim // 2,
                                                dropout=0)(concat_decoded)

        ap_decoded = layers.Reshape((16, 16, 16, 96))(ap_decoded)
        lat_decoded = layers.Reshape((16, 16, 16, 96))(lat_decoded)
        concat_decoded = layers.Reshape((32, 32, 32, 24))(concat_decoded)

    ct_start_channel = 16
    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    init_filter = decode_init_filter
    ct_dim = ct_start_channel
    for index, decode_i in enumerate(range(decode_start_index, 4)):
        current_filter = init_filter // (2 ** decode_i)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        ap_skip_connect = ap_skip_connection_outputs[4 - index]
        ap_skip_connect = SkipUpsample3D(
            current_filter)(ap_skip_connect, ct_dim)
        lat_skip_connect = lat_skip_connection_outputs[4 - index]
        lat_skip_connect = SkipUpsample3D(
            current_filter)(lat_skip_connect, ct_dim)
        ap_decoded = layers.Concatenate(axis=-1)([ap_decoded, ap_skip_connect])
        ap_decoded = HighwayResnetDecoder3D(current_filter,
                                            strides=(2, 2, 2))(ap_decoded)
        lat_decoded = layers.Concatenate(
            axis=-1)([lat_decoded, lat_skip_connect])
        lat_decoded = HighwayResnetDecoder3D(current_filter,
                                             strides=(2, 2, 2))(lat_decoded)

        concat_decoded = layers.Concatenate(
            axis=-1)([ap_decoded, lat_decoded, concat_decoded])
        concat_decoded = HighwayResnetDecoder3D(current_filter,
                                                strides=(2, 2, 2))(concat_decoded)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v3(xray_shape, ct_series_shape,
                             decode_init_filter=64,
                             include_context=False,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
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
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    if include_context:
        # lat_output.shape: [B, 16, 16, 1536]
        _, H, W, C = backend.int_shape(ap_output)
        attn_num_head = 8
        attn_dim = C // attn_num_head
        # concat_decoded.shape: [B, 16, 16, 3072]
        concat_decoded = layers.Concatenate(axis=-1)([ap_output, lat_output])
        # concat_decoded.shape: [B, 32, 32, 768]
        concat_decoded = tf.nn.depth_to_space(concat_decoded, block_size=2)
        concat_decoded = layers.Reshape((H * W * 4, C // 2))(concat_decoded)

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
            concat_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim // 2,
                                                dropout=0)(concat_decoded)

        ap_decoded = layers.Reshape((16, 16, 16, 96))(ap_decoded)
        lat_decoded = layers.Reshape((16, 16, 16, 96))(lat_decoded)
        concat_decoded = layers.Reshape((32, 32, 32, 24))(concat_decoded)

    ct_start_channel = 16
    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    init_filter = decode_init_filter
    ct_dim = ct_start_channel
    for index, decode_i in enumerate(range(decode_start_index, 4)):
        current_filter = init_filter // (2 ** decode_i)

        ap_decoded = conv3d_bn(ap_decoded, current_filter, 3)
        lat_decoded = conv3d_bn(lat_decoded, current_filter, 3)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)

        ap_decoded = Decoder3D(current_filter,
                               strides=(2, 2, 2))(ap_decoded)
        lat_decoded = Decoder3D(current_filter,
                                strides=(2, 2, 2))(lat_decoded)

        concat_decoded = layers.Concatenate(
            axis=-1)([ap_decoded, lat_decoded, concat_decoded])
        concat_decoded = Decoder3D(current_filter,
                                   strides=(2, 2, 2))(concat_decoded)
        concat_decoded = conv3d_bn(concat_decoded, current_filter, 3)
        ct_dim *= 2

    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_channel_activation)(concat_decoded)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v4(xray_shape, ct_series_shape,
                             decode_init_filter=768,
                             skip_connect=True,
                             last_channel_activation="tanh"):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=xray_shape,
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
    )

    if ct_series_shape == (256, 256, 256):
        decode_end_index = 4
    elif ct_series_shape == (128, 128, 128):
        decode_end_index = 3
    elif ct_series_shape == (64, 64, 64):
        decode_end_index = 2
    elif ct_series_shape == (32, 32, 32):
        decode_end_index = 1
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")
    last_channel_num = ct_series_shape[-1]
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]
    # x.shape: [B, 16, 16, 1536]
    x = base_output
    _, H, W, C = backend.int_shape(x)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    x = layers.Reshape((H * W, C))(x)
    x = AddPositionEmbs(input_shape=(H * W, C))(x)
    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        x = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                               dropout=0)(x)
    x = layers.Reshape((H, W, C))(x)

    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(0, decode_end_index)):
        current_filter = init_filter // (2 ** decode_i)
        x = conv2d_bn(x, current_filter, 3)
        if skip_connect:
            skip_connect_output = skip_connection_outputs[4 - index]
            x = layers.Concatenate(axis=-1)([x, skip_connect_output])
        x = Decoder2D(current_filter,
                      kernel_size=2)(x)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(x)
    return Model(base_input, output_tensor)


def get_x2ct_model_ap_lat_v4(xray_shape, ct_series_shape,
                             decode_init_filter=768,
                             skip_connect=True,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
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
        input_shape=(xray_shape[0], xray_shape[1], 1),
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    if ct_series_shape == (256, 256, 256):
        decode_end_index = 4
    elif ct_series_shape == (128, 128, 128):
        decode_end_index = 3
    elif ct_series_shape == (64, 64, 64):
        decode_end_index = 2
    elif ct_series_shape == (32, 32, 32):
        decode_end_index = 1
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")
    last_channel_num = ct_series_shape[-1]
    # x.shape: [B, 16, 16, 1536]

    ap_decoded = ap_output
    lat_decoded = lat_output

    _, H, W, C = backend.int_shape(ap_decoded)
    attn_num_head = 8
    attn_dim = C // attn_num_head

    ap_decoded = layers.Reshape((H * W, C))(ap_decoded)
    ap_decoded = AddPositionEmbs(input_shape=(H * W, C))(ap_decoded)
    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        ap_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=0)(ap_decoded)
    ap_decoded = layers.Reshape((H, W, C))(ap_decoded)

    lat_decoded = layers.Reshape((H * W, C))(lat_decoded)
    lat_decoded = AddPositionEmbs(input_shape=(H * W, C))(lat_decoded)
    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        lat_decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                         dropout=0)(lat_decoded)
    lat_decoded = layers.Reshape((H, W, C))(lat_decoded)

    ct_decode = layers.Concatenate(axis=-1)([ap_decoded, lat_decoded])
    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(0, decode_end_index)):
        current_filter = init_filter // (2 ** decode_i)
        ct_decode = conv2d_bn(ct_decode, current_filter, 3)

        if skip_connect:
            ap_skip_connection_output = ap_skip_connection_outputs[4 - index]
            lat_skip_connection_output = lat_skip_connection_outputs[4 - index]
            ct_decode = layers.Concatenate(axis=-1)([ct_decode,
                                                     ap_skip_connection_output,
                                                     lat_skip_connection_output])
        ct_decode = Decoder2D(current_filter,
                              kernel_size=2)(ct_decode)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(ct_decode)
    return Model([ap_input, lat_input], output_tensor)


def get_x2ct_model_ap_lat_v5(xray_shape, ct_series_shape,
                             block_size=16,
                             decode_init_filter=64,
                             last_channel_activation="tanh"):

    ap_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(xray_shape[0], xray_shape[1], 1),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
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
        input_shape=(xray_shape[0], xray_shape[1], 1),
        block_size=block_size,
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
        name_prefix="lat"
    )
    # lat_input.shape [B, 512, 512, 2]
    # lat_output.shape: [B, 16, 16, 1536]
    lat_input = lat_model.input
    lat_output = lat_model.output
    lat_skip_connection_outputs = [lat_model.get_layer(f"lat_{layer_name}").output
                                   for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    ct_start_channel = 16
    ct_dim = ct_start_channel
    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

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

    ap_decoded = layers.Reshape((16, 16, 16, block_size * 6))(ap_decoded)
    lat_decoded = layers.Reshape((16, 16, 16, block_size * 6))(lat_decoded)
    concat_decoded = (ap_decoded + lat_decoded) / math.sqrt(2)

    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)

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


def get_x2ct_model_ap_lat_v6(xray_shape, ct_series_shape,
                             block_size=16,
                             base_act="relu",
                             dense_middle_filter=256,
                             dense_middle_size=2,
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

    ct_start_channel = 16
    ct_dim = ct_start_channel
    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    ct_start_channel = 16
    ap_decoded = conv2d_bn(ap_output, dense_middle_filter, 3)
    ap_decoded = layers.Flatten()(ap_decoded)
    ap_decoded = layers.Dense(
        ct_start_channel ** 3 * dense_middle_size)(ap_decoded)
    ap_decoded = layers.Reshape(
        (ct_start_channel, ct_start_channel, ct_start_channel, dense_middle_size))(ap_decoded)
    ap_decoded = conv3d_bn(ap_decoded, decode_init_filter, 3)

    lat_decoded = conv2d_bn(lat_output, dense_middle_filter, 3)
    lat_decoded = layers.Flatten()(lat_decoded)
    lat_decoded = layers.Dense(
        ct_start_channel ** 3 * dense_middle_size)(lat_decoded)
    lat_decoded = layers.Reshape(
        (ct_start_channel, ct_start_channel, ct_start_channel, dense_middle_size))(lat_decoded)
    lat_decoded = conv3d_bn(lat_decoded, decode_init_filter, 3)

    concat_decoded = (ap_decoded + lat_decoded) / math.sqrt(2)

    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)

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

    ct_start_channel = 16
    ct_dim = ct_start_channel
    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
    elif ct_series_shape == (64, 64, 64):
        decode_start_index = 3
    elif ct_series_shape == (32, 32, 32):
        decode_start_index = 4
    else:
        NotImplementedError(
            "ct_series_shape is implemented only 128 or 256 intercubic shape")

    ct_start_channel = 16
    init_filter = decode_init_filter
    ap_decoded = SkipUpsample3D(init_filter)(ap_output, ct_dim)
    ap_decoded = conv3d_bn(ap_decoded, decode_init_filter, 3)

    lat_decoded = SkipUpsample3D(init_filter)(lat_output, ct_dim)
    lat_decoded = conv3d_bn(lat_decoded, decode_init_filter, 3)

    concat_decoded = (ap_decoded + lat_decoded) / math.sqrt(2)

    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(decode_start_index, 5)):
        current_filter = init_filter // (2 ** decode_i)

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


def get_ct2x_model(ct_series_shape,
                   block_size=16,
                   dense_middle_filter=256,
                   dense_middle_size=2,
                   decode_init_filter=768,
                   skip_connect=True,
                   base_act="leakyrelu",
                   last_channel_num=1,
                   last_channel_activation="sigmoid"
                   ):

    if ct_series_shape == (256, 256, 256):
        upsample_num = 1
        ct_start_channel = 8
    elif ct_series_shape == (128, 128, 128):
        upsample_num = 2
        ct_start_channel = 4

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

    decoded = conv3d_bn(decoded, dense_middle_filter, 3)
    decoded = layers.Flatten()(decoded)
    decoded = layers.Dense(
        ct_start_channel ** 2 * dense_middle_size)(decoded)
    decoded = layers.Reshape(
        (ct_start_channel, ct_start_channel, dense_middle_size))(decoded)
    decoded = conv2d_bn(decoded, decode_init_filter, 3)
    current_filter = init_filter
    for index, decode_i in enumerate(range(0, 5)):
        if skip_connect:
            skip_connect_output = skip_connection_outputs[4 - index]
            skip_connect_output = backend.mean(skip_connect_output, axis=3)
            skip_connect_output = conv2d_bn(
                skip_connect_output, current_filter, 3)
            decoded = HighwayMulti(dim=current_filter, mode='2d')(
                decoded, skip_connect_output)
        current_filter = init_filter // (2 ** decode_i)
        decoded = conv2d_bn(decoded, current_filter, 3)
        decoded = Decoder2D(current_filter,
                            kernel_size=2)(decoded)

    for index in range(0, upsample_num):

        current_filter = init_filter // (2 ** (decode_i + index + 1))
        decoded = conv2d_bn(decoded, current_filter, 3)
        decoded = Decoder2D(current_filter,
                            kernel_size=2)(decoded)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(decoded)
    return Model(base_input, output_tensor)
