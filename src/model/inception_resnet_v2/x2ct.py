from .base_model import InceptionResNetV2
from .layers import SkipUpsample3D, HighwayResnetDecoder3D, OutputLayer3D
from .layers import inception_resnet_block_3d, conv3d_bn
from tensorflow.keras import Model, layers
from tensorflow.keras import backend
SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "maxpool_2", "mixed_6a", "mixed_7a"]


def get_x2ct_model(xray_shape, ct_series_shape,
                   decode_init_filter=64,
                   include_context=False,
                   last_channel_activation="tanh"):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=xray_shape,
        classes=None,
        padding="same",
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
    # x.shape: [B, 16, 16, 16, 1536]
    x = SkipUpsample3D(filters=1536,
                       include_context=include_context)(base_output, ct_start_channel)
    for block_idx in range(1, 6):
        x = inception_resnet_block_3d(x, scale=0.2,
                                      block_type='block8_3d', block_idx=block_idx)

    if ct_series_shape == (256, 256, 256):
        decode_start_index = 1
    elif ct_series_shape == (128, 128, 128):
        decode_start_index = 2
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
        x = HighwayResnetDecoder3D(current_filter,
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
