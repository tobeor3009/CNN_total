from .base_model import InceptionResNetV2, conv2d_bn, inception_resnet_block
from .layers import HighwayResnetDecoder2D, OutputLayer2D, Decoder2D, AddPositionEmbs, TransformerEncoder
from tensorflow.keras import Model, layers
from tensorflow.keras import backend
SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "maxpool_2", "mixed_6a", "mixed_7a"]


def get_segmentation_model(input_shape,
                           decode_init_filter=768,
                           skip_connect=True,
                           last_channel_num=1,
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
    for index, decode_i in enumerate(range(0, 5)):
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
