from .base_model import conv2d_bn
from .base_model_as_class import InceptionResNetV2_progressive, InceptionResNetV2, InceptionResNetV2_Small
from .base_model_as_class import DecoderBlock2D, DecoderBlock2D_stargan, get_input_label2image_tensor
from .base_model_resnet import HighWayResnet2D, HighWayDecoder2D
from .layers import OutputLayer2D, TwoWayOutputLayer2D, Decoder2D, TransformerEncoder, PatchEmbedding
from .transformer_layers import AddPositionEmbs
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


def get_segmentation_model_v2(input_shape,
                              input_class_num=None,
                              block_size=16,
                              filter_scale=1,
                              groups=1,
                              encode_block="conv",
                              num_downsample=5,
                              use_attention=True,
                              norm="instance",
                              base_act="leakyrelu",
                              last_skip_connect=False,
                              last_act="tanh",
                              last_channel_num=1
                              ):

    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2])
    base_model, skip_connect_layer_names = InceptionResNetV2_progressive(target_shape=target_shape,
                                                                         block_size=block_size,
                                                                         padding="same",
                                                                         groups=groups,
                                                                         norm=norm,
                                                                         base_act=base_act,
                                                                         last_act=base_act,
                                                                         name_prefix="seg",
                                                                         num_downsample=num_downsample,
                                                                         use_attention=use_attention,
                                                                         skip_connect_names=True)

    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    _, H, W, C = base_output.shape

    decoded = base_output

    if input_class_num is not None:
        class_input = layers.Input((input_class_num,))
        class_tensor = layers.Reshape((1, 1, input_class_num))(class_input)
        class_tensor = tf.tile(class_tensor, (1, H, W, 1))
        class_tensor = layers.Conv2D(filters=block_size * 8,
                                     kernel_size=1,
                                     padding="same")(class_tensor)
        C += block_size * 8
        decoded = layers.Concatenate(axis=-1)([decoded, class_tensor])

    if encode_block == "conv":
        pass
    elif encode_block == "transformer":
        attn_dropout_proba = 0.1
        attn_dim_list = [C // 8 for _ in range(6)]
        num_head_list = [8 for _ in range(6)]
        attn_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=attn_dropout_proba)
            attn_layer_list.append(attn_layer)
        attn_sequence = Sequential(attn_layer_list)
        attn_decoded = layers.Reshape((H * W, C))(decoded)
        attn_decoded = AddPositionEmbs(input_shape=(H * W, C))(attn_decoded)
        attn_decoded = attn_sequence(attn_decoded)
        attn_decoded = layers.Reshape((H, W, C))(attn_decoded)
        decoded = layers.Concatenate(axis=-1)([decoded, attn_decoded])

    seg_output = DecoderBlock2D(input_tensor=decoded, encoder=base_model,
                                skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
                                norm=norm, last_channel_num=last_channel_num, filter_scale=filter_scale,
                                groups=groups, num_downsample=num_downsample,
                                base_act=base_act, last_skip_connect=last_skip_connect,
                                last_act=last_act, name_prefix="seg")

    if input_class_num is None:
        return Model(base_input, seg_output)
    else:
        return Model([base_input, class_input], seg_output)


def get_segmentation_model_v3(input_shape,
                              block_size=16,
                              groups=1,
                              encode_block="conv",
                              use_attention=True,
                              norm="instance",
                              base_act="leakyrelu",
                              last_skip_connect=False,
                              last_act="tanh",
                              last_channel_num=1
                              ):

    num_downsample = 3
    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2])

    base_model, skip_connect_layer_names = InceptionResNetV2_Small(target_shape=target_shape,
                                                                   block_size=block_size,
                                                                   padding="same",
                                                                   groups=groups,
                                                                   norm=norm,
                                                                   base_act=base_act,
                                                                   last_act=base_act,
                                                                   name_prefix="seg",
                                                                   use_attention=use_attention,
                                                                   skip_connect_names=True)

    # x.shape: [B, 16, 16, 1536]
    base_input = base_model.input
    base_output = base_model.output
    _, H, W, C = base_output.shape

    decoded = base_output

    if encode_block == "conv":
        pass
    elif encode_block == "transformer":
        attn_dropout_proba = 0
        attn_dim_list = [block_size * 12 for _ in range(6)]
        num_head_list = [8 for _ in range(6)]
        attn_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=attn_dropout_proba)
            attn_layer_list.append(attn_layer)
        attn_sequence = Sequential(attn_layer_list)
        decoded = layers.Reshape((H * W, C))(decoded)
        decoded = PatchEmbedding(num_patch=H * W,
                                 embed_dim=C)(decoded)
        decoded = attn_sequence(decoded)
        decoded = layers.Reshape((H, W, C))(decoded)

    seg_output = DecoderBlock2D(input_tensor=decoded, encoder=base_model,
                                skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
                                norm=norm, last_channel_num=last_channel_num,
                                groups=groups, num_downsample=num_downsample,
                                base_act=base_act, last_skip_connect=last_skip_connect,
                                last_act=last_act, name_prefix="seg")

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
