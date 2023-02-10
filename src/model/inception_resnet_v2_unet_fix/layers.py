from functools import partial
from matplotlib.pyplot import get
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization, SpectralNormalization
from tensorflow.keras import backend
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu, softmax, sigmoid

from .transformer_layers import Attention, AddPositionEmbs

import math
USE_CONV_BIAS = True
USE_DENSE_BIAS = True
GC_BLOCK_RATIO = 0.125
kaiming_initializer = tf.keras.initializers.HeNormal()


def to_kernel_tuple(int_or_list_or_tuple):
    if isinstance(int_or_list_or_tuple, int):
        int_or_list_or_tuple = (int_or_list_or_tuple, int_or_list_or_tuple)
    kernel_tuple = (int_or_list_or_tuple[0],
                    int_or_list_or_tuple[1])
    return kernel_tuple


def to_pad_tuple(int_or_list_or_tuple):
    if isinstance(int_or_list_or_tuple, int):
        int_or_list_or_tuple = (int_or_list_or_tuple, int_or_list_or_tuple)
    pad_tuple = (int_or_list_or_tuple[0] // 2,
                 int_or_list_or_tuple[1] // 2)
    return pad_tuple


def get_norm_layer(norm, axis=-1, name=None):
    if norm == "layer":
        norm_layer = layers.LayerNormalization(axis=axis)
    elif norm == "batch":
        norm_layer = layers.BatchNormalization(axis=axis,
                                               scale=False)
    elif norm == "instance":
        norm_layer = InstanceNormalization(axis=axis)
    elif norm is None:
        def norm_layer(x): return x
    return norm_layer


def get_act_layer(activation, name=None):
    if activation is None:
        def act_layer(x): return x
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6, name=name)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.3, name=name)
    else:
        act_layer = layers.Activation(activation, name=name)
    return act_layer


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(
            input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        # patch.shape = [B num_patch C]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        # embed.shape = [B num_patch embed_dim] + [num_patch]
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed


class EqualizedConv(layers.Layer):
    def __init__(self, out_channels, downsample=False, kernel=3,
                 padding="same", use_bias=True, gain=2, **kwargs):
        super(EqualizedConv, self).__init__(**kwargs)
        if downsample:
            self.strides = 2
        else:
            self.strides = 1
        self.kernel = to_kernel_tuple(kernel)
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.gain = gain
        if padding == "same":
            self.pad = to_pad_tuple(kernel)
        elif padding == "valid":
            self.pad = False

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel[0], self.kernel[1],
                   self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
            )
        fan_in = self.kernel[0] * self.kernel[1] * self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        if isinstance(self.pad, tuple):
            x = tf.pad(inputs, [[0, 0],
                                [self.pad[0], self.pad[0]],
                                [self.pad[1], self.pad[1]],
                                [0, 0]
                                ], mode="REFLECT")
        else:
            x = inputs
        output = tf.nn.conv2d(x, self.scale * self.w, strides=self.strides,
                              padding="VALID")
        if self.use_bias:
            output = output + self.b

        return output


class EqualizedDense(layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier=1, bias_initializer="zeros", **kwargs):
        super(EqualizedDense, self).__init__(**kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer=self.bias_initializer, trainable=True, name="bias"
        )
        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        output = tf.add(tf.matmul(inputs, self.scale * self.w), self.b)
        return output * self.learning_rate_multiplier


class SelfAttention(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not heads == 1

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = layers.Softmax(axis=-1)
        self.to_qkv = EqualizedDense(inner_dim * 3)
        self.to_out = Sequential(
            [
                EqualizedDense(inner_dim),
                layers.Dropout(dropout)
            ]
        ) if project_out else layers.Lambda(lambda x: x)

    def build(self, input_shape):
        super().build(input_shape)
        self.N = input_shape[1]

    def call(self, x):
        # qkv.shape : [B N 3 * dim_head]
        qkv = self.to_qkv(x)
        # qkv.shape : [B N 3 num_heads, dim_head]
        qkv = layers.Reshape((self.N, 3, self.heads, self.dim_head)
                             )(qkv)
        # shape: 3, B, self.num_heads, self.N, (dim_head)
        qkv = backend.permute_dimensions(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q.shape [B num_head N dim]
        # (k.T).shape [B num_head dim N]
        # dots.shape [B num_head N N]
        dots = tf.matmul(q, k, transpose_a=False, transpose_b=True)
        attn = self.attend(dots)
        # attn.shape [B num_head N N]
        # v.shape [B num_head N dim]
        # out.shape [B num_head N dim]
        out = tf.matmul(attn, v)
        # out.shape [B N (num_head * dim)]
        out = layers.Reshape((self.N, self.heads * self.dim_head))(out)
        out = self.to_out(out)
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
        self.ffpn_dense_1 = EqualizedDense(hidden_dim)
        self.ffpn_act_1 = get_act_layer("gelu")
        self.ffpn_dropout_1 = layers.Dropout(dropout)
        self.ffpn_dense_2 = EqualizedDense(self.inner_dim)
        self.ffpn_act_2 = get_act_layer("gelu")
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
        current = self.norm_2(current)

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
        x = get_act_layer(activation, name=ac_name)(x)
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
    def __init__(self, filters, include_context=False, context_head_nums=8,
                 norm="batch", activation="tanh"):
        super().__init__()
        self.include_context = include_context
        compress_layer_list = [
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            get_norm_layer(norm),
            get_act_layer(activation)
        ]
        if self.include_context == True:
            up_head_dim = filters // context_head_nums
            self.context_layer = TransformerEncoder2D(heads=8, dim_head=up_head_dim,
                                                      dropout=0.3)
        self.compress_block = Sequential(compress_layer_list)
        self.conv_block = Sequential([
            layers.Conv3D(filters, kernel_size=3, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            get_norm_layer(norm),
            get_act_layer(activation)
        ])

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, input_tensor, Z):
        conv = self.compress_block(input_tensor)
        if self.include_context == True:
            conv = self.context_layer(conv, self.H, self.W)
        # shape: [B H W 1 C]
        conv = backend.expand_dims(conv, axis=1)
        # shape: [B H W Z C]
        conv = backend.repeat_elements(conv, rep=Z, axis=1)
        conv = self.conv_block(conv)
        return conv


class Pixelshuffle3D(layers.Layer):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = self.to_tuple(kernel_size)

    # k: kernel, r: resized, o: original
    def call(self, x):
        _, o_h, o_w, o_z, o_c = backend.int_shape(x)
        k_h, k_w, k_z = self.kernel_size
        r_h, r_w, r_z = o_h * k_h, o_w * k_w, o_z * k_z
        r_c = o_c // (k_h * k_w * k_z)

        r_x = layers.Reshape((o_h, o_w, o_z, r_c, k_h, k_w, k_z))(x)
        r_x = layers.Permute((1, 5, 2, 6, 3, 7, 4))(r_x)
        r_x = layers.Reshape((r_h, r_w, r_z, r_c))(r_x)

        return r_x

    def compute_output_shape(self, input_shape):
        _, o_h, o_w, o_z, o_c = input_shape
        k_h, k_w, k_z = self.kernel_size
        r_h, r_w, r_z = o_h * k_h, o_w * k_w, o_z * k_z
        r_c = o_c // (r_h * r_w * r_z)

        return (r_h, r_w, r_z, r_c)

    def to_tuple(self, int_or_tuple):
        if isinstance(int_or_tuple, int):
            convert_tuple = (int_or_tuple, int_or_tuple, int_or_tuple)
        else:
            convert_tuple = int_or_tuple
        return convert_tuple


class HighwayMulti(layers.Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, dim, mode='3d', transform_gate_bias=-3, name=None, **kwargs):
        super(HighwayMulti, self).__init__(**kwargs)
        self.mode = mode
        self.transform_gate_bias = transform_gate_bias
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dim = dim
        self.output_name = name
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
        value = layers.Add(name=self.output_name)(
            [transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(HighwayMulti, self).get_config()
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


class HighwayDecoder2D(layers.Layer):
    def __init__(self, filters, strides, kernel_size=2, activation="tanh", unsharp=False):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.conv_before_pixel_shffle = layers.Conv2D(filters=filters,
                                                      kernel_size=1, padding="same",
                                                      strides=1, use_bias=USE_CONV_BIAS)
        self.conv_after_pixel_shffle = layers.Conv2D(filters=filters,
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
        self.act_layer = get_act_layer(activation)
        self.highway_layer = HighwayMulti(dim=filters, mode='2d')

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(filters)

    def call(self, input_tensor):

        pixel_shffle = self.conv_before_pixel_shffle(input_tensor)
        pixel_shffle = tf.nn.depth_to_space(pixel_shffle,
                                            block_size=self.kernel_size)
        pixel_shffle = self.conv_after_pixel_shffle(pixel_shffle)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(pixel_shffle, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)

        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class PixelShuffleBlock2D(layers.Layer):
    def __init__(self, out_channel, kernel_size=2,
                 norm="batch", activation="leakyrelu", unsharp=False):
        super(PixelShuffleBlock2D, self).__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.conv_after_pixel_shffle = layers.Conv2D(filters=out_channel,
                                                     kernel_size=3, padding="same",
                                                     strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel * 2)

    def call(self, input_tensor):

        pixel_shuffle = tf.nn.depth_to_space(input_tensor,
                                             block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        output = self.norm_layer(pixel_shuffle)
        output = self.act_layer(output)
        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class SELNorm(layers.Layer):

    def call(self, x, heatmap):
        x_mean, x_std = self.get_mean_std(x)
        heatmap_mean, heatmap_std = self.get_mean_std(heatmap)
        valid_normalized = (x - x_mean) / x_std
        train_normalized = heatmap_std * valid_normalized + heatmap_mean
        return backend.in_test_phase(train_normalized,
                                     valid_normalized)

    def get_mean_std(self, x, epsilon=1e-5):
        axes = [1, 2]
        # Compute the mean and standard deviation of a tensor.
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + epsilon)

        return mean, standard_deviation


class Decoder2D(layers.Layer):
    def __init__(self, out_channel, kernel_size=2,
                 norm="batch", activation="leakyrelu", unsharp=False):
        super(Decoder2D, self).__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.conv_before_pixel_shffle = layers.Conv2D(filters=out_channel * (kernel_size ** 2),
                                                      kernel_size=1, padding="same",
                                                      strides=1, use_bias=USE_CONV_BIAS)
        self.conv_after_pixel_shffle = layers.Conv2D(filters=out_channel,
                                                     kernel_size=3, padding="same",
                                                     strides=1, use_bias=USE_CONV_BIAS)

        self.upsample_layer = layers.UpSampling2D(
            size=kernel_size, interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=out_channel,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel * 2)

    def call(self, input_tensor):

        pixel_shuffle = self.conv_before_pixel_shffle(input_tensor)
        pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle,
                                             block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)

        output = layers.Concatenate()([pixel_shuffle, upsample])
        output = self.norm_layer(output)
        output = self.act_layer(output)
        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class Decoder2DHeatmap(layers.Layer):
    def __init__(self, out_channel, kernel_size=2,
                 norm="batch", activation="leakyrelu", unsharp=False):
        super(Decoder2DHeatmap, self).__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.conv_before_pixel_shffle = layers.Conv2D(filters=out_channel * (kernel_size ** 2),
                                                      kernel_size=1, padding="same",
                                                      strides=1, use_bias=USE_CONV_BIAS)
        self.conv_after_pixel_shffle = layers.Conv2D(filters=out_channel,
                                                     kernel_size=3, padding="same",
                                                     strides=1, use_bias=USE_CONV_BIAS)

        self.upsample_layer = layers.UpSampling2D(size=kernel_size,
                                                  interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=out_channel,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.norm_layer = SELNorm()
        self.act_layer = get_act_layer(activation)

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel * 2)

    def call(self, input_tensor, heatmap):

        pixel_shuffle = self.conv_before_pixel_shffle(input_tensor)
        pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle,
                                             block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)

        output = layers.Concatenate()([pixel_shuffle, upsample])
        output = self.norm_layer(output, heatmap)
        output = self.act_layer(output)
        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class UpsampleBlock2D(layers.Layer):
    def __init__(self, out_channel, kernel_size=2,
                 norm="batch", activation="leakyrelu", unsharp=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.upsample_layer = layers.UpSampling2D(size=kernel_size,
                                                  interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=out_channel,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.norm_layer_upsample = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel)

    def call(self, input_tensor):
        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)
        upsample = self.norm_layer_upsample(upsample)
        output = self.act_layer(upsample)
        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class UpsampleBlock2DHeatmap(layers.Layer):
    def __init__(self, out_channel, kernel_size=2,
                 norm="batch", activation="leakyrelu", unsharp=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.upsample_layer = layers.UpSampling2D(
            size=kernel_size, interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=out_channel,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.norm_layer_upsample = SELNorm()
        self.act_layer = get_act_layer(activation)

        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel)

    def call(self, input_tensor, heatmap):
        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)
        upsample = self.norm_layer_upsample(upsample, heatmap)
        output = self.act_layer(upsample)
        if self.unsharp is True:
            output = self.unsharp_mask_layer(output)

        return output


class HighwayDecoder3D(layers.Layer):
    def __init__(self, filters, strides, activation):
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
        self.act_layer = get_act_layer(activation)
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
    def __init__(self, filters, strides, norm, activation):
        super().__init__()

        self.filters = filters
        self.pixel_shuffle = Pixelshuffle3D(kernel_size=2)
        self.conv_after_pixel_shuffle = layers.Conv3D(filters=filters,
                                                      kernel_size=3, padding="same",
                                                      strides=1, use_bias=False)

        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=False)

        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):

        pixel_shuffle = self.pixel_shuffle(input_tensor)
        pixel_shuffle = self.conv_after_pixel_shuffle(pixel_shuffle)

        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)

        output = layers.Concatenate()([pixel_shuffle,
                                       upsample])
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class PixelShuffleBlock3D(layers.Layer):
    def __init__(self, filters, strides, norm, activation):
        super().__init__()

        self.filters = filters
        self.conv_before_pixel_shuffle = layers.Conv3D(filters=filters * 8,
                                                       kernel_size=3, padding="same",
                                                       strides=1, use_bias=False)
        self.pixel_shuffle = Pixelshuffle3D(kernel_size=strides)

        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):

        pixel_shuffle = self.conv_before_pixel_shuffle(input_tensor)
        pixel_shuffle = self.pixel_shuffle(pixel_shuffle)

        output = self.norm_layer(pixel_shuffle)
        output = self.act_layer(output)
        return output


class UpsampleBlock3D(layers.Layer):
    def __init__(self, filters, strides, norm, activation):
        super().__init__()

        self.filters = filters
        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=1, use_bias=False)

        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):

        upsample = self.upsample_layer(input_tensor)
        upsample = self.conv_after_upsample(upsample)
        upsample = self.norm_layer(upsample)
        output = self.act_layer(upsample)
        return output


class TransposeBlock3D(layers.Layer):
    def __init__(self, filters, strides, norm, activation):
        super().__init__()

        self.filters = filters
        self.conv_transpose = layers.Conv3DTranspose(filters=filters,
                                                     kernel_size=3, padding="same",
                                                     strides=strides, use_bias=False)

        self.norm_layer = get_norm_layer(norm)
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):

        transpose = self.conv_transpose(input_tensor)
        transpose = self.norm_layer(transpose)
        output = self.act_layer(transpose)
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
        self.act = get_act_layer(act)

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
        self.act = get_act_layer(act)

    def call(self, x1, x2):
        conv_1 = self.conv_1(x1)
        conv_2 = self.conv_2(x2)
        output = self.highway_layer(conv_1, conv_2)
        output = self.act(output)

        return output


class SimpleOutputLayer2D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_4x4 = layers.Conv2D(filters=last_channel_num,
                                      kernel_size=4,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.act = get_act_layer(act)

    def call(self, input_tensor):
        conv_4x4 = self.conv_4x4(input_tensor)
        output = self.act(conv_4x4)

        return output


class OutputLayer3D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_1x1x1 = layers.Conv3D(filters=last_channel_num,
                                        kernel_size=1,
                                        padding="same",
                                        strides=1,
                                        use_bias=False,
                                        )
        self.conv_3x3x3 = layers.Conv3D(filters=last_channel_num,
                                        kernel_size=3,
                                        padding="same",
                                        strides=1,
                                        use_bias=False,
                                        )
        self.highway_layer = HighwayMulti(dim=last_channel_num, mode='3d')
        self.act = get_act_layer(act)

    def call(self, input_tensor):
        conv_1x1x1 = self.conv_1x1x1(input_tensor)
        conv_3x3x3 = self.conv_3x3x3(input_tensor)

        output = self.highway_layer(conv_1x1x1, conv_3x3x3)
        output = self.act(output)

        return output


class SimpleOutputLayer3D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        self.conv_4x4x4 = layers.Conv3D(filters=last_channel_num,
                                        kernel_size=4,
                                        padding="same",
                                        strides=1,
                                        use_bias=False,
                                        )
        self.act = get_act_layer(act)

    def call(self, input_tensor):
        conv_4x4x4 = self.conv_4x4x4(input_tensor)
        output = self.act(conv_4x4x4)

        return output
