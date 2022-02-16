from .base_model import SegInceptionResNetV2
from .layers import SkipUpsample3D, HighwayResnetDecoder3D, OutputLayer
from .layers import inception_resnet_block_3d, conv3d_bn
from tensorflow.keras import Model, layers
from tensorflow.keras import backend
SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "maxpool_2", "mixed_6a", "mixed_7a"]


def get_x2ct_model(xray_shape, ct_series_shape,
                   decode_init_filter=64,
                   include_context=False,
                   last_channel_activation="tanh"):

    base_model = SegInceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=xray_shape,
        classes=None,
        pooling=None,
        classifier_activation=None,
        include_context=include_context,
    )
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    ct_start_channel = 16
    # x.shape: [B, 16, 16, 16, 1536]
    x = SkipUpsample3D(filters=1536, in_channel=1536,
                       include_context=include_context)(base_output, ct_start_channel)
    for block_idx in range(1, 6):
        x = inception_resnet_block_3d(x, scale=0.17,
                                      block_type='block35_3d', block_idx=block_idx,
                                      include_context=include_context)
    for block_idx in range(1, 6):
        x = inception_resnet_block_3d(x, scale=0.1,
                                      block_type='block17_3d', block_idx=block_idx,
                                      include_context=include_context)
    for block_idx in range(1, 6):
        x = inception_resnet_block_3d(x, scale=0.2,
                                      block_type='block8_3d', block_idx=block_idx,
                                      include_context=include_context)

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
        x = conv3d_bn(x, current_filter, 3, include_context=include_context)
        skip_connect = skip_connection_outputs[4 - index]
        skip_connect_channel = backend.int_shape(skip_connect)[-1]
        skip_connect = SkipUpsample3D(current_filter,
                                      in_channel=skip_connect_channel,
                                      include_context=include_context)(skip_connect, ct_dim)
        x = layers.Concatenate(axis=-1)([x, skip_connect])
        x = HighwayResnetDecoder3D(current_filter,
                                   strides=(2, 2, 2))(x)
        ct_dim *= 2

    output_tensor = OutputLayer(last_channel_num=1,
                                act=last_channel_activation)(x)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(base_input, output_tensor)
