from torch import pixel_shuffle
from .base_model import InceptionResNetV2, conv2d_bn, inception_resnet_block
from .layers import HighwayResnetDecoder2D, OutputLayer2D, Decoder2D, TransformerEncoder
from .transformer_layers import AddPositionEmbs, PosEncodingLayer, AddPosEncoding
from .reformer_layers import ReformerBlock
from tensorflow.keras import Model, layers
from tensorflow.keras import backend
SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "maxpool_2", "mixed_6a", "mixed_7a"]


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
                           decode_init_filter=768,
                           skip_connect=True,
                           last_channel_num=1,
                           last_channel_activation="tanh",
                           version="transformer"):

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
    init_filter = decode_init_filter
    x = base_output
    _, H, W, C = backend.int_shape(x)
    attn_num_head = 8
    attn_dim = C // attn_num_head
    x = layers.Reshape((H * W, C))(x)
    # x = AddPositionEmbs(input_shape=(H * W, C))(x)
    x = AddPosEncoding()(x)
    if version == "transformer":

        attn_dim_list = [attn_dim for _ in range(6)]
        num_head_list = [attn_num_head for _ in range(6)]
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            x = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                   dropout=0)(x)
        x = layers.Reshape((H, W, C))(x)

        pixel_shuffle = conv2d_bn(x, init_filter, 3)
        upsample = conv2d_bn(x, init_filter, 3)

    if version == "reformer":
        x1 = x
        x2 = x
        attn_dim_list = [attn_dim for _ in range(6)]
        num_head_list = [attn_num_head for _ in range(6)]
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            inner_dim = attn_dim * num_head
            x1, x2 = ReformerBlock(d_model=inner_dim, d_ff=inner_dim, max_len=H * W,
                                   attn_config=AttnConfig(inner_dim, num_head))(x1, x2, t=H * W)
        x = layers.Reshape((H, W, C))(x)

        pixel_shuffle = conv2d_bn(x1, init_filter, 3)
        upsample = conv2d_bn(x2, init_filter, 3)

    for index, decode_i in enumerate(range(0, 5)):
        if skip_connect:
            skip_connect_output = skip_connection_outputs[4 - index]
            pixel_shuffle = layers.Concatenate(
                axis=-1)([pixel_shuffle, skip_connect_output])
            upsample = layers.Concatenate(
                axis=-1)([upsample, skip_connect_output])
        current_filter = init_filter // (2 ** decode_i)
        pixel_shuffle = conv2d_bn(pixel_shuffle, current_filter, 3)
        upsample = conv2d_bn(upsample, current_filter, 3)
        pixel_shuffle, upsample = Decoder2D(current_filter,
                                            kernel_size=2)(pixel_shuffle, upsample)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_channel_activation)(pixel_shuffle, upsample)
    return Model(base_input, output_tensor)
