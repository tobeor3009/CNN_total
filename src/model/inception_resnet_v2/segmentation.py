from .base_model import InceptionResNetV2, conv2d_bn, inception_resnet_block
from .layers import HighwayResnetDecoder2D, OutputLayer2D
from tensorflow.keras import Model, layers
from tensorflow.keras import backend
SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "maxpool_2", "mixed_6a", "mixed_7a"]


def get_segmentation_model(input_shape,
                           decode_init_filter=1536,
                           include_context=False,
                           last_channel_activation="tanh"):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        classes=None,
        padding="same",
        pooling=None,
        classifier_activation=None,
    )
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]

    # x.shape: [B, 16, 16, 1536]
    for block_idx in range(1, 6):
        x = inception_resnet_block(x, scale=0.2,
                                   block_type='block8', block_idx=block_idx,
                                   include_context=include_context)

    init_filter = decode_init_filter
    for index, decode_i in enumerate(range(0, 5)):
        current_filter = init_filter // (2 ** decode_i)
        x = conv2d_bn(x, current_filter, 3,
                      include_context=include_context)
        skip_connect = skip_connection_outputs[4 - index]
        x = layers.Concatenate(axis=-1)([x, skip_connect])
        x = HighwayResnetDecoder2D(current_filter,
                                   strides=(2, 2))(x)

    output_tensor = OutputLayer2D(last_channel_num=1,
                                  act=last_channel_activation)(x)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    return Model(base_input, output_tensor)
