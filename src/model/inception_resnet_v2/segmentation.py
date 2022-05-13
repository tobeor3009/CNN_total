from .base_model import InceptionResNetV2, SKIP_CONNECTION_LAYER_NAMES
from .base_model import conv2d_bn
from .base_model_resnet import HighWayResnet2D, HighWayDecoder2D
from .layers import OutputLayer2D, TwoWayOutputLayer2D, Decoder2D, TransformerEncoder
from .transformer_layers import AddPositionEmbs, PosEncodingLayer, AddPosEncoding
from .reformer_layers import ReformerBlock
from tensorflow.keras import Model, layers
from tensorflow.keras import backend


class AttnConfig():
    def __init__(self, dim, num_heads):
        self.dim = dim
        self.num_heads = num_heads
        self.num_hashes = 2
        self.bucket_size = 8
        self.causality = False
        self.causal_start = None
        self.use_full = False


def get_segmentation_model(input_shape,
                           block_size=16,
                           decode_init_filter=768,
                           skip_connect=True,
                           base_act="leakyrelu",
                           last_channel_num=1,
                           last_channel_activation="tanh"
                           ):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        block_size=block_size,
        classes=None,
        padding="same",
        base_act=base_act,
        last_act=base_act,
        pooling=None,
        classifier_activation=None,
    )
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]
    init_filter = decode_init_filter
    decoded = base_output

    for _ in range(3):
        decoded = conv2d_bn(decoded, init_filter, 3)
    for index, decode_i in enumerate(range(0, 5)):
        if skip_connect:
            skip_connect_output = skip_connection_outputs[4 - index]
            decoded = layers.Concatenate(axis=-1)([decoded,
                                                   skip_connect_output])
        current_filter = init_filter // (2 ** decode_i)
        decoded = conv2d_bn(decoded, current_filter, 3)
        decoded = Decoder2D(current_filter,
                            kernel_size=2)(decoded)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(decoded)
    return Model(base_input, output_tensor)


def get_resnet_segmentation_model(input_shape,
                                  block_size=16,
                                  decode_init_filter=768,
                                  skip_connect=True,
                                  base_act="leakyrelu",
                                  last_channel_num=1,
                                  last_channel_activation="tanh"
                                  ):

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        block_size=block_size,
        classes=None,
        padding="same",
        base_act=base_act,
        last_act=base_act,
        pooling=None,
        classifier_activation=None,
    )
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    skip_connection_outputs = [base_model.get_layer(layer_name).output
                               for layer_name in SKIP_CONNECTION_LAYER_NAMES]
    init_filter = decode_init_filter
    decoded = base_output

    for _ in range(3):
        decoded = conv2d_bn(decoded, init_filter, 3)
    for index, decode_i in enumerate(range(0, 5)):
        if skip_connect:
            skip_connect_output = skip_connection_outputs[4 - index]
            decoded = layers.Concatenate(axis=-1)([decoded,
                                                   skip_connect_output])
        current_filter = init_filter // (2 ** decode_i)
        decoded = conv2d_bn(decoded, current_filter, 3)
        decoded = Decoder2D(current_filter,
                            kernel_size=2)(decoded)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(decoded)
    return Model(base_input, output_tensor)


def get_highway_resnet_model(input_shape, last_channel_num, block_size=16,
                             encoder_output_filter=None,
                             groups=1, num_downsample=5,
                             base_act="relu", last_act="tanh"):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    ################################################
    ################# Define call ##################
    ################################################
    input_tensor = encoder.input
    encoder_output = encoder(input_tensor)

    seg_output = HighWayDecoder2D(input_tensor=encoder_output, encoder=encoder,
                                  skip_connection_layer_names=SKIP_CONNECTION_LAYER_NAMES,
                                  last_filter=last_channel_num,
                                  block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act=last_act, name_prefix="seg")
    model = Model(input_tensor, seg_output)
    return model
