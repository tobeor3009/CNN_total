from .base_model import conv2d_bn
from .base_model_as_class import InceptionResNetV2_progressive, InceptionResNetV2, InceptionResNetV2_Stargan
from .base_model_as_class import DecoderBlock2D, DecoderBlock2D_stargan, get_input_label2image_tensor
from .base_model_resnet import HighWayResnet2D, HighWayDecoder2D
from .layers import OutputLayer2D, TwoWayOutputLayer2D, Decoder2D, TransformerEncoder
from .transformer_layers import AddPositionEmbs, PosEncodingLayer, AddPosEncoding
from .reformer_layers import ReformerBlock
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
USE_CONV_BIAS = True
USE_DENSE_BIAS = True


class AttnConfig():
    def __init__(self, dim, num_heads):
        self.dim = dim
        self.num_heads = num_heads
        self.num_hashes = 2
        self.bucket_size = 8
        self.causality = False
        self.causal_start = None
        self.use_full = False


def get_segmentation_model_v1(input_shape,
                              block_size=16,
                              encode_block="conv",
                              base_act="leakyrelu",
                              last_act="tanh",
                              last_channel_num=1
                              ):

    base_model = InceptionResNetV2(input_shape=input_shape,
                                   block_size=block_size,
                                   padding="same",
                                   base_act=base_act,
                                   last_act=base_act,
                                   name_prefix="seg",
                                   use_attention=True)
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    B, H, W, C = base_output.shape

    init_filter = C
    decoded = base_output

    if encode_block == "conv":
        pass
    elif encode_block == "transformer":
        attn_dropout_proba = 0.1
        attn_dim_list = [block_size * 3 for _ in range(6)]
        num_head_list = [8 for _ in range(6)]
        attn_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=attn_dropout_proba)
            attn_layer_list.append(attn_layer)
        attn_sequence = Sequential(attn_layer_list)

        decoded = attn_sequence(decoded)
        decoded = tf.reshape(decoded, (B, H, W, C))

    for _ in range(3):
        decoded = conv2d_bn(decoded, init_filter, 3,
                            activation=base_act)

    for idx in range(5, 0, -1):
        skip_connect = base_model.get_layer(f"seg_down_block_{idx}").output
        _, _, _, current_filter = skip_connect.shape
        decoded = layers.Concatenate(axis=-1)([decoded,
                                               skip_connect])
        decoded = conv2d_bn(decoded, current_filter, 3,
                            activation=base_act)
        decoded = Decoder2D(current_filter,
                            kernel_size=2,
                            activation=base_act)(decoded)

    output_tensor = OutputLayer2D(last_channel_num=last_channel_num,
                                  act=last_act)(decoded)
    return Model(base_input, output_tensor)

def get_segmentation_model_v2(input_shape,
                              block_size=16,
                              groups=1,
                              encode_block="conv",
                              base_act="leakyrelu",
                              last_act="tanh",
                              last_channel_num=1
                              ):

    base_model, skip_connect_layer_names = InceptionResNetV2(input_shape=input_shape,
                                                             block_size=block_size,
                                                             padding="same",
                                                             groups=groups,
                                                             base_act=base_act,
                                                             last_act=base_act,
                                                             name_prefix="seg",
                                                             use_attention=True,
                                                             skip_connect_names=True)
    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    B, H, W, C = base_output.shape

    init_filter = C
    decoded = base_output

    if encode_block == "conv":
        pass
    elif encode_block == "transformer":
        attn_dropout_proba = 0.1
        attn_dim_list = [block_size * 3 for _ in range(6)]
        num_head_list = [8 for _ in range(6)]
        attn_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=attn_dropout_proba)
            attn_layer_list.append(attn_layer)
        attn_sequence = Sequential(attn_layer_list)

        decoded = attn_sequence(decoded)
        decoded = tf.reshape(decoded, (B, H, W, C))

    seg_output = DecoderBlock2D(input_tensor=decoded, encoder=base_model,
                                skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
                                last_channel_num=last_channel_num,
                                groups=groups, num_downsample=5,
                                base_act=base_act, last_act=last_act, name_prefix="seg")

    return Model(base_input, seg_output)

def get_segmentation_model_stargan(input_shape,
                                   label_len,
                                   block_size=16,
                                   groups=1,
                                   base_act="leakyrelu",
                                   last_act="tanh",
                                   last_channel_num=1
                                   ):

    label_added_shape = (input_shape[0], input_shape[1],
                         input_shape[2] + label_len)
    base_model, skip_connect_layer_names = InceptionResNetV2(input_shape=label_added_shape,
                                                             block_size=block_size,
                                                             padding="same",
                                                             groups=groups,
                                                             base_act=base_act,
                                                             last_act=base_act,
                                                             name_prefix="seg",
                                                             use_attention=True,
                                                             skip_connect_names=True)
    B, H, W, C = base_model.output.shape

    input_label_shape = (input_shape[0],
                         input_shape[1],
                         label_len)

    target_label_shape = (input_shape[0] // (2 ** 5),
                          input_shape[1] // (2 ** 5),
                          C)
    class_input, class_tensor = get_input_label2image_tensor(label_len=label_len, target_shape=input_label_shape,
                                                             dropout_ratio=0.2, reduce_level=5)
    target_class_input, target_class_tensor = get_input_label2image_tensor(label_len=label_len, target_shape=target_label_shape,
                                                                           dropout_ratio=0.2, reduce_level=2)
    model_input = layers.Input(input_shape)
    model_concat_input = layers.Concatenate(axis=-1)([model_input,
                                                      class_tensor])

    decoded = base_model(model_concat_input)

    seg_output = DecoderBlock2D_stargan(input_tensor=decoded, label_tensor=target_class_tensor, encoder=base_model,
                                        skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
                                        last_channel_num=last_channel_num,
                                        block_size=block_size, groups=groups, num_downsample=5,
                                        base_act=base_act, last_act=last_act, name_prefix="seg")

    return Model([model_input, class_input, target_class_input], seg_output)


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
