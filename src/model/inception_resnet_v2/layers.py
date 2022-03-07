from functools import partial
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu, softmax, sigmoid

from .transformer_layers import SelfAttention, Attention, AddPositionEmbs

import math
USE_CONV_BIAS = True
USE_DENSE_BIAS = True
GC_BLOCK_RATIO = 0.125
kaiming_initializer = tf.keras.initializers.HeNormal()


class GCBlock2D(layers.Layer):
    def __init__(self, in_channel, ratio=GC_BLOCK_RATIO, fusion_types=('channel_add',), **kwargs):
        super().__init__(**kwargs)
        assert in_channel is not None, 'GCBlock needs in_channel'
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.ratio = ratio
        self.middle_channel = max(int(in_channel * ratio), 2)
        self.fusion_types = fusion_types
        self.key_mask = layers.Conv2D(filters=self.middle_channel,
                                      kernel_size=1,
                                      kernel_initializer=kaiming_initializer,
                                      padding="same",
                                      use_bias=USE_CONV_BIAS)
        self.value_mask = layers.Conv2D(filters=1,
                                        kernel_size=1,
                                        kernel_initializer=kaiming_initializer,
                                        padding="same",
                                        use_bias=USE_CONV_BIAS)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_mul_conv = None

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, x):

        # x.shape: [B, H, W, C]
        # key_mask.shape: [B, H, W, middle_channel]
        key_mask = self.key_mask(x)
        # key_mask.shape: [B, (H * W), middle_channel]
        key_mask = layers.Reshape(
            (self.H * self.W, self.middle_channel))(key_mask)
        # key_mask.shape: [B, middle_channel, (H * W)]
        key_mask = layers.Permute((2, 1))(key_mask)

        # value_mask.shape: [B, H, W, 1]
        value_mask = self.value_mask(x)
        # value_mask.shape: [B, (H * W), 1]
        value_mask = layers.Reshape((self.H * self.W, 1))(value_mask)
        value_mask = self.softmax(value_mask)

        # [B, middle_channel, (H * W)] @ [B, (H * W), 1]
        # context_mask.shape: [B, middle_channel, 1]
        context_mask = tf.matmul(key_mask, value_mask)
        # context_mask.shape: [B, 1, 1, middle_channel]
        context_mask = layers.Reshape(
            (1, 1, self.middle_channel))(context_mask)

        out = x
        if self.channel_mul_conv is not None:
            # [B, 1, 1, C]
            channel_mul_term = sigmoid(self.channel_mul_conv(context_mask))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [B, 1, 1, C]
            channel_add_term = self.channel_add_conv(context_mask)
            out = out + channel_add_term
        # out.shape: [B, H, W, C]
        return out


class GCBlock3D(layers.Layer):
    def __init__(self, in_channel, ratio=GC_BLOCK_RATIO, fusion_types=('channel_add',), **kwargs):
        super().__init__(**kwargs)
        assert in_channel is not None, 'GCBlock needs in_channel'
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.ratio = ratio
        self.middle_channel = max(int(in_channel * ratio), 2)
        self.fusion_types = fusion_types

        self.key_mask = layers.Conv3D(filters=self.middle_channel,
                                      kernel_size=1,
                                      kernel_initializer=kaiming_initializer,
                                      padding="same",
                                      use_bias=USE_CONV_BIAS)
        self.value_mask = layers.Conv3D(filters=1,
                                        kernel_size=1,
                                        kernel_initializer=kaiming_initializer,
                                        padding="same",
                                        use_bias=USE_CONV_BIAS)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_mul_conv = None

    def build(self, input_shape):
        _, self.H, self.W, self.Z, self.C = input_shape

    def call(self, x):

        # x.shape: [B, H, W, Z, C]
        # key_mask.shape: [B, H, W, Z, C]
        key_mask = self.key_mask(x)
        # key_mask.shape: [B, (H * W * Z), C]
        key_mask = layers.Reshape(
            (self.H * self.W * self.Z, self.middle_channel))(key_mask)
        # key_mask.shape: [B, C, (H * W)]
        key_mask = layers.Permute((2, 1))(key_mask)

        # value_mask.shape: [B, H, W, Z, 1]
        value_mask = self.value_mask(x)
        # value_mask.shape: [B, (H * W * Z), 1]
        value_mask = layers.Reshape((self.H * self.W * self.Z, 1))(value_mask)
        value_mask = self.softmax(value_mask)

        # [B, C, (H * W * Z)] @ [B, (H * W * Z), 1]
        # context_mask.shape: [B, C, 1]
        context_mask = tf.matmul(key_mask, value_mask)
        # context_mask.shape: [B, 1, 1, 1, C]
        context_mask = layers.Reshape(
            (1, 1, 1, self.middle_channel))(context_mask)

        out = x
        if self.channel_mul_conv is not None:
            # [B, 1, 1, 1, C]
            channel_mul_term = sigmoid(self.channel_mul_conv(context_mask))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [B, 1, 1, 1, C]
            channel_add_term = self.channel_add_conv(context_mask)
            out = out + channel_add_term
        # out.shape: [B, H, W, Z, C]
        return out


class TransformerEncoder(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 hidden_dim=2048, dropout: float = 0.):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.self_attn = SelfAttention(heads, dim_head, dropout)
        self.self_attn_dropout = layers.Dropout(dropout)
        self.self_attn_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.ffpn_dense_1 = layers.Dense(hidden_dim, use_bias=False)
        self.ffpn_act_1 = layers.Activation(tf.nn.relu6)
        self.ffpn_dropout_1 = layers.Dropout(dropout)
        self.ffpn_dense_2 = layers.Dense(self.inner_dim, use_bias=False)
        self.ffpn_act_2 = layers.Activation(tf.nn.relu6)
        self.ffpn_dropout_2 = layers.Dropout(dropout)
        self.ffpn_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, x):
        self_attn = self.self_attn(x)
        self_attn = self.self_attn_dropout(self_attn)
        self_attn = self.self_attn_norm(x + self_attn)

        out = self.ffpn_dense_1(self_attn)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(self_attn + out)

        return out


class TransformerEncoder2D(TransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, H, W):
        x = layers.Reshape((H * W, self.inner_dim))(x)
        self_attn = self.self_attn(x)
        self_attn = self.self_attn_dropout(self_attn)
        self_attn = self.self_attn_norm(x + self_attn)

        out = self.ffpn_dense_1(self_attn)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(self_attn + out)
        out = layers.Reshape((H, W, self.inner_dim))(out)
        return out


class TransformerEncoder3D(TransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, H, W, Z):
        x = layers.Reshape((H * W * Z, self.inner_dim))(x)
        self_attn = self.self_attn(x)
        self_attn = self.self_attn_dropout(self_attn)
        self_attn = self.self_attn_norm(x + self_attn)

        out = self.ffpn_dense_1(self_attn)
        out = self.ffpn_dropout_1(out)
        out = self.ffpn_act_1(out)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_act_2(out)
        out = self.ffpn_dropout_2(out)
        out = self.ffpn_norm(self_attn + out)

        out = layers.Reshape((H, W, Z, self.inner_dim))(out)
        return out


class TransformerDecoder(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 hidden_dim=2048, dropout: float = 0.):

        super().__init__()
        self.inner_dim = heads * dim_head
        self.self_attn = SelfAttention(heads, dim_head, dropout)
        self.self_attn_dropout = layers.Dropout(dropout)
        self.norm_1 = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.attn = Attention(heads, dim_head, dropout)
        self.attn_dropout = layers.Dropout(dropout)
        self.norm_2 = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.attn_dense_1 = layers.Dense(hidden_dim, use_bias=False)
        self.attn_act_1 = layers.Activation(tf.nn.relu6)
        self.attn_dropout_1 = layers.Dropout(dropout)
        self.attn_dense_2 = layers.Dense(self.inner_dim, use_bias=False)
        self.attn_dropout_2 = layers.Dropout(dropout)
        self.norm_3 = layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, current, hidden):
        self_attn = self.self_attn(current)
        self_attn = self.self_attn_dropout(self_attn)

        current = current + self_attn
        current = self.norm_1(current)

        attn = self.attn(current, hidden)
        attn = self.attn_dropout(attn)
        current = current + attn
        current = self.norm2(current)

        attn = self.attn_dense_1(current)
        attn = self.attn_act_1(attn)
        attn = self.attn_dropout_1(attn)
        attn = self.attn_dense_2(attn)
        attn = self.attn_dropout_2(attn)
        current = current + attn
        current = self.norm_3(current)
        return current


def get_transformer_layer(x, num_layer: int = 6, heads: int = 8, dim_head: int = 64,
                          hidden_dim: int = 2048, dropout: float = 0.):

    encoded_tensor = x
    encoder_tensor_shape = backend.int_shape(encoded_tensor)
    encoded_tensor = AddPositionEmbs(
        input_shape=encoder_tensor_shape)(encoded_tensor)

    for _ in range(num_layer):
        encoded_tensor = TransformerEncoder(heads=heads, dim_head=dim_head,
                                            hidden_dim=hidden_dim, dropout=dropout)(encoded_tensor)

    encoded_tensor = layers.LayerNormalization(axis=-1,
                                               epsilon=1e-6)(encoded_tensor)
    decoded_tensor = encoded_tensor
    encoded_tensor = AddPositionEmbs(
        input_shape=encoder_tensor_shape)(encoded_tensor)

    for _ in range(num_layer):
        decoded_tensor = TransformerDecoder(heads=heads, dim_head=dim_head,
                                            hidden_dim=hidden_dim,
                                            dropout=dropout)(decoded_tensor, encoded_tensor)

    decoded_tensor = layers.LayerNormalization(axis=-1,
                                               epsilon=1e-6)(decoded_tensor)
    return decoded_tensor


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              include_context=False,
              context_head_nums=8,
              name=None):
    """Utility function to apply conv + BN.

    Args:
      x: input tensor.
      filters: filters in `Conv2D`.
      kernel_size: kernel size as in `Conv2D`.
      strides: strides in `Conv2D`.
      padding: padding mode in `Conv2D`.
      activation: activation in `Conv2D`.
      use_bias: whether to use a bias in `Conv2D`.
      name: name of the ops; will become `name + '_ac'` for the activation
          and `name + '_bn'` for the batch norm layer.

    Returns:
      Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv3D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name)(
            x)
    if not use_bias:
        bn_axis = -1
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=ac_name)(x)
        else:
            x = layers.Activation(activation, name=ac_name)(x)
    if include_context == True:
        context_shape = backend.int_shape(x)
        context_head_dim = context_shape[-1] // context_head_nums
        context = TransformerEncoder3D(heads=context_head_nums, dim_head=context_head_dim,
                                       dropout=0.3)(x, *context_shape[1:-1])
        return x + context
    else:
        return x


def inception_resnet_block_3d(x, scale, block_type, block_idx, activation='relu',
                              include_context=False, context_head_nums=8):
    """Adds an Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`

    Args:
      x: input tensor.
      scale: scaling factor to scale the residuals (i.e., the output of passing
        `x` through an inception module) before adding them to the shortcut
        branch. Let `r` be the output from the residual branch, the output of this
        block will be `x + scale * r`.
      block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
        structure in the residual branch.
      block_idx: an `int` used for generating layer names. The Inception-ResNet
        blocks are repeated many times in this network. We use `block_idx` to
        identify each of the repetitions. For example, the first
        Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
        and the layer names will have a common prefix `'block35_0'`.
      activation: activation function to use at the end of the block (see
        [activations](../activations.md)). When `activation=None`, no activation
        is applied
        (i.e., "linear" activation: `a(x) = x`).

    Returns:
        Output tensor for the block.

    Raises:
      ValueError: if `block_type` is not one of `'block35'`,
        `'block17'` or `'block8'`.
    """
    if block_type == 'block35_3d':
        branch_0 = conv3d_bn(x, 160, 1)
        branch_1 = conv3d_bn(x, 160, 1)
        branch_1 = conv3d_bn(branch_1, 160, 3)
        branch_2 = conv3d_bn(x, 160, 1)
        branch_2 = conv3d_bn(branch_2, 240, 3)
        branch_2 = conv3d_bn(branch_2, 320, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17_3d':
        branch_0 = conv3d_bn(x, 288, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 240, [1, 1, 7])
        branch_1 = conv3d_bn(branch_1, 264, [1, 7, 1])
        branch_1 = conv3d_bn(branch_1, 288, [7, 1, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8_3d':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 224, [1, 1, 3])
        branch_1 = conv3d_bn(branch_1, 240, [1, 3, 1])
        branch_1 = conv3d_bn(branch_1, 256, [3, 1, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35_3d", "block17_3d" or "branch_1", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = -1
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(
            branches)
    up = conv3d_bn(
        mixed,
        backend.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_name + '_conv')
    if include_context == True:
        up_shape = backend.int_shape(up)
        up_head_dim = up_shape[-1] // context_head_nums
        up = TransformerEncoder3D(heads=context_head_nums, dim_head=up_head_dim,
                                  dropout=0.3)(up, *up_shape[1:-1])
    x = layers.Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=backend.int_shape(x)[1:],
        arguments={'scale': scale},
        name=block_name)([x, up])
    if activation is not None:
        if activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=block_name + '_ac')(x)
        else:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


class SkipUpsample3D(layers.Layer):
    def __init__(self, filters, include_context=False, context_head_nums=8):
        super().__init__()
        self.include_context = include_context
        compress_layer_list = [
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ]
        if self.include_context == True:
            up_head_dim = filters // context_head_nums
            self.context_layer = TransformerEncoder2D(heads=8, dim_head=up_head_dim,
                                                      dropout=0.3)
        self.compress_block = Sequential(compress_layer_list)
        self.conv_block = Sequential([
            layers.Conv3D(filters, kernel_size=3, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ])

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, input_tensor, Z):
        conv = self.compress_block(input_tensor)
        if self.include_context == True:
            conv = self.context_layer(conv, self.H, self.W)
        # shape: [B H W 1 C]
        conv = backend.expand_dims(conv, axis=-2)
        # shape: [B H W Z C]
        conv = backend.repeat_elements(conv, rep=Z, axis=-2)
        conv = self.conv_block(conv)
        return conv


class SkipUpsample3D(layers.Layer):
    def __init__(self, filters, include_context=False, context_head_nums=8):
        super().__init__()
        self.include_context = include_context
        compress_layer_list = [
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ]
        if self.include_context == True:
            up_head_dim = filters // context_head_nums
            self.context_layer = TransformerEncoder2D(heads=8, dim_head=up_head_dim,
                                                      dropout=0.3)
        self.compress_block = Sequential(compress_layer_list)
        self.conv_block = Sequential([
            layers.Conv3D(filters, kernel_size=3, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ])

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, input_tensor, Z):
        conv = self.compress_block(input_tensor)
        if self.include_context == True:
            conv = self.context_layer(conv, self.H, self.W)
        # shape: [B H W 1 C]
        conv = backend.expand_dims(conv, axis=-2)
        # shape: [B H W Z C]
        conv = backend.repeat_elements(conv, rep=Z, axis=-2)
        conv = self.conv_block(conv)
        return conv


# class SkipUpsample3D(layers.Layer):
#     def __init__(self, filters, context_head_nums=8):
#         super().__init__()
#         self.filters = filters
#         compress_layer_list = [
#             layers.Conv2D(filters, kernel_size=1, padding="same",
#                           strides=1, use_bias=USE_CONV_BIAS),
#             layers.BatchNormalization(axis=-1),
#             layers.Activation("tanh")
#         ]
#         up_head_dim = filters // context_head_nums
#         self.context_layer = TransformerEncoder2D(heads=context_head_nums, dim_head=up_head_dim,
#                                                   dropout=0)

#         self.compress_block = Sequential(compress_layer_list)
#         self.conv_block = Sequential([
#             layers.Conv3D(filters, kernel_size=3, padding="same",
#                           strides=1, use_bias=USE_CONV_BIAS),
#             layers.BatchNormalization(axis=-1),
#             layers.Activation("tanh")
#         ])

#     def build(self, input_shape):
#         _, self.H, self.W, self.C = input_shape

#     def call(self, input_tensor, H, W, Z):
#         conv = self.compress_block(input_tensor)
#         conv = self.context_layer(conv, H, W)
#         conv = layers.Reshape((H, W, Z, self.filters // Z))(conv)
#         # shape: [B H W Z C]
#         conv = self.conv_block(conv)
#         return conv


class HighwayMulti(layers.Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, dim, mode='3d', activation='relu', transform_gate_bias=-3, **kwargs):
        super(HighwayMulti, self).__init__(**kwargs)
        self.mode = mode
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dim = dim
        self.dense_1 = layers.Dense(units=self.dim,
                                    use_bias=USE_DENSE_BIAS, bias_initializer=transform_gate_bias_initializer)

    def call(self, x, y):
        if self.mode == '2d':
            transform_gate = layers.GlobalAveragePooling2D()(x)
        elif self.mode == '3d':
            transform_gate = layers.GlobalAveragePooling3D()(x)
        transform_gate = self.dense_1(transform_gate)
        transform_gate = layers.Activation("sigmoid")(transform_gate)
        carry_gate = layers.Lambda(lambda x: 1.0 - x,
                                   output_shape=(self.dim,))(transform_gate)
        transformed_gated = layers.Multiply()([transform_gate, x])
        identity_gated = layers.Multiply()([carry_gate, y])
        value = layers.Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(HighwayMulti, self).get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config


def get_gaussian_kernel(size=2, mean=0.0, std=1.0):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.compat.v1.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


class UnsharpMasking2D(layers.Layer):
    def __init__(self, filters):
        super(UnsharpMasking2D, self).__init__()
        gauss_kernel_2d = get_gaussian_kernel(2, 0.0, 1.0)
        self.gauss_kernel = tf.tile(
            gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, filters, 1])

        self.pointwise_filter = tf.eye(filters, batch_shape=[1, 1])

    def call(self, input_tensor):
        blur_tensor = tf.nn.separable_conv2d(input_tensor,
                                             self.gauss_kernel,
                                             self.pointwise_filter,
                                             strides=[1, 1, 1, 1], padding='SAME')
        unsharp_mask_tensor = 2 * input_tensor - blur_tensor
        # because it used after tanh
        unsharp_mask_tensor = tf.clip_by_value(unsharp_mask_tensor, -1, 1)
        return unsharp_mask_tensor


class HighwayResnetDecoder2D(layers.Layer):
    def __init__(self, filters, strides):
        super().__init__()

        self.filters = filters
        self.conv_before_trans = layers.Conv2D(filters=filters,
                                               kernel_size=1, padding="same",
                                               strides=1, use_bias=USE_CONV_BIAS)
        self.conv_trans = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=strides, use_bias=USE_CONV_BIAS)
        self.conv_after_trans = layers.Conv2D(filters=filters,
                                              kernel_size=1, padding="same",
                                              strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv2D(filters=filters,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)
        self.upsample_layer = layers.UpSampling2D(size=strides)
        self.conv_after_upsample = layers.Conv2D(filters=filters,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters, mode='2d')

    def call(self, input_tensor):

        conv_trans = self.conv_before_trans(input_tensor)
        conv_trans = self.conv_trans(conv_trans)
        conv_trans = self.conv_after_trans(conv_trans)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(conv_trans, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class Decoder2D(layers.Layer):
    def __init__(self, out_channel, kernel_size=2, unsharp=False):
        super(Decoder2D, self).__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.conv_before_pixel_shffle = layers.Conv2D(filters=out_channel * (kernel_size ** 2),
                                                      kernel_size=1, padding="same",
                                                      strides=1, use_bias=USE_CONV_BIAS)
        self.conv_after_pixel_shffle = layers.Conv2D(filters=out_channel,
                                                     kernel_size=1, padding="same",
                                                     strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv2D(filters=out_channel,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)

        self.upsample_layer = layers.UpSampling2D(
            size=kernel_size, interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=out_channel,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.pixel_shffle_input_control_conv = layers.Conv2DTranspose(filters=out_channel,
                                                                      kernel_size=1, padding="same",
                                                                      strides=2, use_bias=USE_CONV_BIAS)
        self.upsample_input_control_conv = layers.Conv2DTranspose(filters=out_channel,
                                                                  kernel_size=1, padding="same",
                                                                  strides=2, use_bias=USE_CONV_BIAS)

        self.norm_layer_pixel_shffle = layers.LayerNormalization(axis=-1)
        self.norm_layer_upsample = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel)

    def call(self, pixel_shuffle_input, upsample_input):

        pixel_shuffle = self.conv_before_pixel_shffle(pixel_shuffle_input)
        pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle,
                                             block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)
        pixel_shuffle = self.norm_layer_pixel_shffle(pixel_shuffle)

        upsample = self.conv_before_upsample(upsample_input)
        upsample = self.upsample_layer(upsample)
        upsample = self.conv_after_upsample(upsample)
        upsample = self.norm_layer_upsample(upsample)

        pixel_shuffle_input_resized = self.pixel_shffle_input_control_conv(
            pixel_shuffle_input)
        upsample_input_resized = self.upsample_input_control_conv(
            upsample_input)

        pixel_shuffle_output = (
            pixel_shuffle_input_resized + upsample) / math.sqrt(2)
        upsample_output = (upsample_input_resized +
                           pixel_shuffle) / math.sqrt(2)

        if self.unsharp is True:
            pixel_shuffle_output = self.unsharp_mask_layer(
                pixel_shuffle_output)
            upsample_output = self.unsharp_mask_layer(upsample_output)

        return pixel_shuffle_output, upsample_output


class HighwayResnetDecoder3D(layers.Layer):
    def __init__(self, filters, strides):
        super().__init__()

        self.filters = filters
        self.conv_before_trans = layers.Conv3D(filters=filters,
                                               kernel_size=1, padding="same",
                                               strides=1, use_bias=USE_CONV_BIAS)
        self.conv_trans = layers.Conv3DTranspose(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=strides, use_bias=USE_CONV_BIAS)
        self.conv_after_trans = layers.Conv3D(filters=filters,
                                              kernel_size=1, padding="same",
                                              strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv3D(filters=filters,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)
        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters, mode='3d')

    def call(self, input_tensor):

        conv_trans = self.conv_before_trans(input_tensor)
        conv_trans = self.conv_trans(conv_trans)
        conv_trans = self.conv_after_trans(conv_trans)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(conv_trans, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class Decoder3D(layers.Layer):
    def __init__(self, filters, strides):
        super().__init__()

        self.filters = filters
        self.conv_before_trans = layers.Conv3D(filters=filters,
                                               kernel_size=1, padding="same",
                                               strides=1, use_bias=USE_CONV_BIAS)
        self.conv_trans = layers.Conv3DTranspose(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=strides, use_bias=USE_CONV_BIAS)
        self.conv_after_trans = layers.Conv3D(filters=filters,
                                              kernel_size=1, padding="same",
                                              strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv3D(filters=filters,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)
        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer_conv = layers.LayerNormalization(axis=-1)
        self.norm_layer_upsample = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh

    def call(self, input_tensor):

        conv_trans = self.conv_before_trans(input_tensor)
        conv_trans = self.conv_trans(conv_trans)
        conv_trans = self.conv_after_trans(conv_trans)
        conv_trans = self.norm_layer_conv(conv_trans)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)
        upsamle = self.norm_layer_upsample(upsamle)

        output = (conv_trans + upsamle) / math.sqrt(2)
        output = self.act_layer(output)
        return output


class OutputLayer2D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_1x1 = layers.Conv2D(filters=last_channel_num,
                                      kernel_size=1,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.conv_3x3 = layers.Conv2D(filters=last_channel_num,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.highway_layer = HighwayMulti(dim=last_channel_num, mode='2d')
        self.act = layers.Activation(act)

    def call(self, input_tensor):
        conv_1x1 = self.conv_1x1(input_tensor)
        conv_3x3 = self.conv_3x3(input_tensor)
        output = self.highway_layer(conv_1x1, conv_3x3)
        output = self.act(output)

        return output


class TwoWayOutputLayer2D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_1 = layers.Conv2D(filters=last_channel_num,
                                    kernel_size=3,
                                    padding="same",
                                    strides=1,
                                    use_bias=USE_CONV_BIAS,
                                    )
        self.conv_2 = layers.Conv2D(filters=last_channel_num,
                                    kernel_size=3,
                                    padding="same",
                                    strides=1,
                                    use_bias=USE_CONV_BIAS,
                                    )
        self.highway_layer = HighwayMulti(dim=last_channel_num, mode='2d')
        self.act = layers.Activation(act)

    def call(self, x1, x2):
        conv_1 = self.conv_1(x1)
        conv_2 = self.conv_2(x2)
        output = self.highway_layer(conv_1, conv_2)
        output = self.act(output)

        return output


class OutputLayer3D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_1x1x1 = layers.Conv3D(filters=last_channel_num,
                                        kernel_size=1,
                                        padding="same",
                                        strides=1,
                                        use_bias=USE_CONV_BIAS,
                                        )
        self.conv_3x3x3 = layers.Conv3D(filters=last_channel_num,
                                        kernel_size=3,
                                        padding="same",
                                        strides=1,
                                        use_bias=USE_CONV_BIAS,
                                        )
        self.highway_layer = HighwayMulti(dim=last_channel_num, mode='3d')
        self.act = layers.Activation(act)

    def call(self, input_tensor):
        conv_1x1x1 = self.conv_1x1x1(input_tensor)
        conv_3x3x3 = self.conv_3x3x3(input_tensor)

        output = self.highway_layer(conv_1x1x1, conv_3x3x3)
        output = self.act(output)

        return output
