
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import softmax

from .util_layers import drop_path, get_norm_layer, get_act_layer


def get_len(x):
    if hasattr(x, "shape"):
        x_len = x.shape[0]
    else:
        x_len = len(x)
    return x_len


def meshgrid_3d(*arrs):
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(get_len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


def window_partition(x, window_size):

    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, H, W, C = x.get_shape().as_list()
    # Subset tensors to patches
    window_size_h = window_size[0]
    window_size_w = window_size[1]
    patch_num_h = H // window_size_h
    patch_num_w = W // window_size_w
    x = tf.reshape(x, shape=(-1, patch_num_h, window_size_h,
                   patch_num_w, window_size_w, C))
    # (-1, patch_num_H, patch_num_W, window_size, window_size, C)
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))

    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, window_size_h, window_size_w, C))

    return windows


def window_partition_3d(x, window_size):

    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, Z, H, W, C = x.get_shape().as_list()
    # Subset tensors to patches
    window_size_z = window_size[0]
    window_size_h = window_size[1]
    window_size_w = window_size[2]
    patch_num_z = Z // window_size_z
    patch_num_h = H // window_size_h
    patch_num_w = W // window_size_w
    x = tf.reshape(x, shape=(-1, patch_num_z, window_size_z,
                   patch_num_h, window_size_h,
                   patch_num_w, window_size_w, C))
    x = tf.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))

    # Reshape patches to a patch sequence
    windows = tf.reshape(x,
                         shape=(-1, window_size_z, window_size_h, window_size_w, C))

    return windows


def window_reverse(windows, window_size, H, W, C):

    # Reshape a patch sequence to aligned patched
    window_size_h = window_size[0]
    window_size_w = window_size[1]
    patch_num_h = H // window_size_h
    patch_num_w = W // window_size_w
    x = tf.reshape(windows, shape=(-1, patch_num_h, patch_num_w,
                                   window_size_h, window_size_w, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))

    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))

    return x


def window_reverse_3d(windows, window_size, Z, H, W, C):

    # Reshape a patch sequence to aligned patched
    window_size_z = window_size[0]
    window_size_h = window_size[1]
    window_size_w = window_size[2]
    patch_num_z = Z // window_size_z
    patch_num_h = H // window_size_h
    patch_num_w = W // window_size_w
    x = tf.reshape(windows, shape=(-1, patch_num_z, patch_num_h, patch_num_w,
                                   window_size_z, window_size_h, window_size_w, C))
    x = tf.transpose(x, perm=(0, 1, 4, 2, 5, 3, 6, 7))

    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, Z, H, W, C))

    return x


class Mlp(layers.Layer):
    def __init__(self, filter_num, act="gelu", drop=0., name=''):

        super().__init__()

        # MLP layers
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))

        # Dropout layer
        self.drop = Dropout(drop)

        # default: GELU activation
        self.activation = get_act_layer(act)

    def call(self, x):

        # MLP --> GELU --> Drop --> MLP --> Drop
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.window_size = window_size  # size of the attention window
        self.num_heads = num_heads  # number of self-attention heads
        self.qkv_bias = not swin_v2 and qkv_bias
        self.qk_scale = qk_scale
        self.swin_v2 = swin_v2
        self.prefix = name

        # Layers
        self.qkv = Dense(dim * 3, use_bias=self.qkv_bias,
                         name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        C = input_shape[-1]
        # zero initialization
        num_window_elements = ((2 * self.window_size[0] - 1) *
                               (2 * self.window_size[1] - 1))
        if self.swin_v2:
            self.scale = self.add_weight(
                'logit_scale',
                shape=[self.num_heads, 1, 1],
                initializer=initializers.Constant(np.log(10.)),
                trainable=True,
                dtype=self.dtype)
            self.cpb0 = layers.Dense(512,
                                     activation='relu',
                                     name='cpb_mlp.0')
            self.cpb1 = layers.Dense(self.num_heads,
                                     activation='sigmoid', use_bias=False,
                                     name='cpb_mlp.2')

            self.q_bias = None
            self.v_bias = None
            if self.qkv_bias:
                self.q_bias = self.add_weight('q_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
                self.v_bias = self.add_weight('v_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
        else:
            head_dim = self.dim // self.num_heads
            self.scale = self.qk_scale or head_dim ** -0.5  # query scaling factor
            self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                                shape=(num_window_elements,
                                                                       self.num_heads),
                                                                initializer=tf.initializers.Zeros(), trainable=True)

        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='{}_attn_pos_ind'.format(self.prefix)
        )

        self.built = True

    def call(self, x, mask=None):

        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C // self.num_heads

        x_qkv = self.qkv(x)

        if self.swin_v2 and self.qkv_bias:
            k_bias = tf.zeros_like(self.v_bias, self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            x_qkv = tf.nn.bias_add(x_qkv, qkv_bias)

        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(x_qkv, 3)

        if self.swin_v2:
            scale = tf.minimum(self.scale, np.log(1. / .01))
            scale = tf.exp(scale)
            q, _ = tf.linalg.normalize(q, axis=-1)
            k, _ = tf.linalg.normalize(k, axis=-1)
        else:
            scale = self.scale

        # Query rescaling
        q = q * self.scale

        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)
        # Shift window
        num_window_elements = np.prod(self.window_size)
        relative_position_index_flat = tf.reshape(self.relative_position_index,
                                                  shape=(-1,))
        if self.swin_v2:
            relative_bias = self.cpb0(self.relative_table(self.window_size))
            relative_bias = self.cpb1(relative_bias)
            relative_bias = tf.reshape(relative_bias, [-1, self.num_heads])
            bias = tf.gather(relative_bias, relative_position_index_flat) * 16.
        else:
            bias = tf.gather(self.relative_position_bias_table,
                             relative_position_index_flat)

        bias = tf.reshape(bias,
                          shape=(num_window_elements, num_window_elements, -1))
        bias = tf.transpose(bias,
                            perm=(2, 0, 1))
        attn = attn + bias[None]

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                                 tf.float32)
            attn = tf.reshape(
                attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)

        # Dropout after attention
        attn = self.attn_drop(attn)

        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))

        # Linear projection
        x_qkv = self.proj(x_qkv)

        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)

        return x_qkv

    def relative_table(self, window_size):
        offset0 = tf.range(1 - window_size[0], window_size[0])
        offset0 = tf.cast(offset0, self.compute_dtype)
        offset1 = tf.range(1 - window_size[1], window_size[1])
        offset1 = tf.cast(offset1, self.compute_dtype)
        # offset.shape = [2 np.prod(window_size) np.prod(window_size))]
        offset = tf.stack(tf.meshgrid(offset0, offset1, indexing='ij'),
                          axis=0)
        # offset.shape = [1 np.prod(window_size), np.prod(window_size) 2]
        offset = tf.transpose(offset, [1, 2, 0])[None]

        window = window_size
        offset *= 8. / (tf.cast(window, self.compute_dtype)[None] - 1.)
        offset = tf.sign(offset) * tf.math.log1p(tf.abs(offset)) / np.log(8)

        return offset


class ContextWindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.window_size = window_size  # size of the attention window
        self.num_heads = num_heads  # number of self-attention heads
        self.qkv_bias = not swin_v2 and qkv_bias
        self.qk_scale = qk_scale
        self.swin_v2 = swin_v2
        self.prefix = name

        # Layers
        self.dense_q = Dense(dim, use_bias=self.qkv_bias,
                             name='{}_attn_q'.format(self.prefix))
        self.dense_kv = Dense(dim * 2, use_bias=self.qkv_bias,
                              name='{}_attn_kv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        C = input_shape[-1]
        # zero initialization
        num_window_elements = ((2 * self.window_size[0] - 1) *
                               (2 * self.window_size[1] - 1))
        if self.swin_v2:
            self.scale = self.add_weight(
                'logit_scale',
                shape=[self.num_heads, 1, 1],
                initializer=initializers.Constant(np.log(10.)),
                trainable=True,
                dtype=self.dtype)
            self.cpb0 = layers.Dense(512,
                                     activation='relu',
                                     name='cpb_mlp.0')
            self.cpb1 = layers.Dense(self.num_heads,
                                     activation='sigmoid', use_bias=False,
                                     name='cpb_mlp.2')

            self.q_bias = None
            self.v_bias = None
            if self.qkv_bias:
                self.q_bias = self.add_weight('q_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
                self.v_bias = self.add_weight('v_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
        else:
            head_dim = self.dim // self.num_heads
            self.scale = self.qk_scale or head_dim ** -0.5  # query scaling factor
            self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                                shape=(num_window_elements,
                                                                       self.num_heads),
                                                                initializer=tf.initializers.Zeros(), trainable=True)

        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='{}_attn_pos_ind'.format(self.prefix)
        )

        self.built = True

    def call(self, x, y, mask=None):

        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C // self.num_heads

        x_q = self.dense_q(x)
        x_kv = self.dense_kv(y)
        if self.swin_v2 and self.qkv_bias:
            k_bias = tf.zeros_like(self.v_bias, self.compute_dtype)
            kv_bias = tf.concat([k_bias, self.v_bias], axis=0)
            x_q = tf.nn.bias_add(x_q, self.q_bias)
            x_kv = tf.nn.bias_add(x_kv, kv_bias)
        x_q = tf.reshape(x_q, shape=(-1, N, self.num_heads, head_dim))
        q = tf.transpose(x_q, perm=(0, 2, 1, 3))
        x_kv = tf.reshape(x_kv, shape=(-1, N, 2, self.num_heads, head_dim))
        x_kv = tf.transpose(x_kv, perm=(2, 0, 3, 1, 4))
        k, v = tf.unstack(x_kv, 2)

        if self.swin_v2:
            scale = tf.minimum(self.scale, np.log(1. / .01))
            scale = tf.exp(scale)
            q, _ = tf.linalg.normalize(q, axis=-1)
            k, _ = tf.linalg.normalize(k, axis=-1)
        else:
            scale = self.scale
        # Query rescaling
        q = q * self.scale
        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)
        # Shift window
        num_window_elements = np.prod(self.window_size)
        relative_position_index_flat = tf.reshape(self.relative_position_index,
                                                  shape=(-1,))
        if self.swin_v2:
            relative_bias = self.cpb0(self.relative_table(self.window_size))
            relative_bias = self.cpb1(relative_bias)
            relative_bias = tf.reshape(relative_bias, [-1, self.num_heads])
            bias = tf.gather(relative_bias, relative_position_index_flat) * 16.
        else:
            bias = tf.gather(self.relative_position_bias_table,
                             relative_position_index_flat)

        bias = tf.reshape(bias,
                          shape=(num_window_elements, num_window_elements, -1))
        bias = tf.transpose(bias,
                            perm=(2, 0, 1))
        attn = attn + bias[None]

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                                 tf.float32)
            attn = tf.reshape(
                attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)

        # Dropout after attention
        attn = self.attn_drop(attn)

        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))

        # Linear projection
        x_qkv = self.proj(x_qkv)

        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)

        return x_qkv

    def relative_table(self, window_size):
        offset0 = tf.range(1 - window_size[0], window_size[0])
        offset0 = tf.cast(offset0, self.compute_dtype)
        offset1 = tf.range(1 - window_size[1], window_size[1])
        offset1 = tf.cast(offset1, self.compute_dtype)
        # offset.shape = [2 np.prod(window_size) np.prod(window_size))]
        offset = tf.stack(tf.meshgrid(offset0, offset1, indexing='ij'),
                          axis=0)
        # offset.shape = [1 np.prod(window_size), np.prod(window_size) 2]
        offset = tf.transpose(offset, [1, 2, 0])[None]

        window = window_size
        offset *= 8. / (tf.cast(window, self.compute_dtype)[None] - 1.)
        offset = tf.sign(offset) * tf.math.log1p(tf.abs(offset)) / np.log(8)

        return offset


class WindowAttention3D(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.window_size = window_size  # size of the attention window
        self.num_heads = num_heads  # number of self-attention heads
        self.qkv_bias = not swin_v2 and qkv_bias
        self.qk_scale = qk_scale
        self.swin_v2 = swin_v2
        self.prefix = name

        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        C = input_shape[-1]

        # zero initialization
        num_window_elements = np.prod((
            2 * self.window_size[0] - 1,
            2 * self.window_size[1] - 1,
            2 * self.window_size[2] - 1
        ))
        if self.swin_v2:
            self.scale = self.add_weight(
                'logit_scale',
                shape=[self.num_heads, 1, 1],
                initializer=initializers.Constant(np.log(10.)),
                trainable=True,
                dtype=self.dtype)
            self.cpb0 = layers.Dense(512,
                                     activation='relu',
                                     name='cpb_mlp.0')
            self.cpb1 = layers.Dense(self.num_heads,
                                     activation='sigmoid', use_bias=False,
                                     name='cpb_mlp.2')
            self.q_bias = None
            self.v_bias = None
            if self.qkv_bias:
                self.q_bias = self.add_weight('q_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
                self.v_bias = self.add_weight('v_bias',
                                              shape=[C],
                                              initializer='zeros',
                                              trainable=True,
                                              dtype=self.dtype)
        else:
            head_dim = self.dim // self.num_heads
            self.scale = self.qk_scale or head_dim ** -0.5
            self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                                shape=(
                                                                    num_window_elements, self.num_heads),
                                                                initializer=tf.initializers.Zeros(), trainable=True)

        # Indices of relative positions
        coords_z = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        coords_matrix = meshgrid_3d(coords_z, coords_h, coords_w)
        coords = np.stack(coords_matrix)
        # coords_flatten.shape = [3 window_size window_size window_size]
        coords_flatten = coords.reshape(3, -1)
        # relative_coords.shape = [3 window_size ** 3 window_size ** 3]
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        # relative_coords.shape = [window_size ** 3 window_size ** 3 3]
        relative_coords = relative_coords.transpose([1, 2, 0])

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1) *
                                     (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        # relative_position_index.shape = [8 8]
        relative_position_index = relative_coords.sum(-1)
        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='{}_attn_pos_ind'.format(self.prefix)
        )

        self.built = True

    def call(self, x, mask=None):

        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C // self.num_heads

        # x_qkv.shape = [B, embed_dim * 3, C]
        x_qkv = self.qkv(x)

        if self.swin_v2 and self.qkv_bias:
            k_bias = tf.zeros_like(self.v_bias, self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            x_qkv = tf.nn.bias_add(x_qkv, qkv_bias)

        # x_qkv.shape = [B, N, 3, num_heads, head_dim]
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        # x_qkv.shape = [3, B, num_heads, N, head_dim]
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        # q.shape = k.shape = v.shape = [B, num_heads, N, head_dim]
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        if self.swin_v2:
            scale = tf.minimum(self.scale, np.log(1. / .01))
            scale = tf.exp(scale)
            q, _ = tf.linalg.normalize(q, axis=-1)
            k, _ = tf.linalg.normalize(k, axis=-1)
        else:
            scale = self.scale
        # Query rescaling
        q = q * self.scale

        # multi-headed self-attention
        # k.shape = [B, num_heads, head_dim, N]
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        # attn.shape = [B, num_heads, N, N]
        attn = (q @ k)

        # Shift window
        num_window_elements = np.prod(self.window_size)
        relative_position_index_flat = tf.reshape(self.relative_position_index,
                                                  shape=(-1,))
        if self.swin_v2:
            relative_bias = self.cpb0(self.relative_table(self.window_size))
            relative_bias = self.cpb1(relative_bias)
            relative_bias = tf.reshape(relative_bias, [-1, self.num_heads])
            bias = tf.gather(relative_bias, relative_position_index_flat) * 16
        else:
            bias = tf.gather(self.relative_position_bias_table,
                             relative_position_index_flat)

        bias = tf.reshape(bias,
                          shape=(num_window_elements, num_window_elements, -1))
        bias = tf.transpose(bias,
                            perm=(2, 0, 1))

        attn = attn + bias[None]

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                                 tf.float32)
            attn = tf.reshape(
                attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)

        # Dropout after attention
        attn = self.attn_drop(attn)

        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))

        # Linear projection
        x_qkv = self.proj(x_qkv)

        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)

        return x_qkv

    def relative_table(self, window_size):
        offset0 = tuple(range(1 - window_size[0], window_size[0]))
        offset1 = tuple(range(1 - window_size[1], window_size[1]))
        offset2 = tuple(range(1 - window_size[2], window_size[2]))
        # offset.shape = [3 np.prod(window_size), np.prod(window_size))]
        offset = tf.stack(meshgrid_3d(offset0, offset1, offset2),
                          axis=0)
        offset = tf.cast(offset, self.compute_dtype)
        # offset.shape = [1 np.prod(window_size), np.prod(window_size) 3]
        offset = tf.transpose(offset, [1, 2, 3, 0])[None]

        window = window_size
        offset *= 8. / (tf.cast(window, self.compute_dtype)[None] - 1.)
        offset = tf.sign(offset) * tf.math.log1p(tf.abs(offset)) / np.log(8)

        return offset


class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, act="gelu", norm="layer",
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        # number of embedded patches; a tuple of  (heigh, width)
        self.num_patch = num_patch
        self.num_heads = num_heads  # number of attention heads
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)  # size of window
        else:
            self.window_size = window_size
        if isinstance(shift_size, int):
            self.shift_size = (shift_size,
                               shift_size)  # size of window shift
        else:
            self.shift_size = shift_size
        self.num_mlp = num_mlp  # number of MLP nodes
        self.prefix = name

        norm = None if swin_v2 else norm
        # Layers
        self.norm1 = get_norm_layer(norm, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=proj_drop,
                                    swin_v2=swin_v2, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = get_norm_layer(norm, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], act=act,
                       drop=mlp_drop, name=self.prefix)

        # Assertions
        for shift_size in self.shift_size:
            assert 0 <= shift_size, 'shift_size >= 0 is required'
        for shift_size, window_size in zip(self.shift_size, self.window_size):
            assert shift_size < window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < min(self.window_size):
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if np.prod(self.shift_size) > 0:
            H, W = self.num_patch
            num_window_elements = np.prod(self.window_size)
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))

            # attention mask
            mask_array = np.zeros((1, H, W, 1))

            # initialization
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows,
                                      shape=[-1, num_window_elements])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask,
                                         trainable=False,
                                         name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        num_window_elements = np.prod(self.window_size)
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)

        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if np.max(self.shift_size) > 0:
            shifted_x = tf.roll(x,
                                shift=[-self.shift_size[0],
                                       -self.shift_size[1]],
                                axis=[1, 2])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows,
                               shape=(-1, num_window_elements, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows,
                                  shape=(-1, *self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if np.max(self.shift_size) > 0:
            x = tf.roll(shifted_x,
                        shift=[self.shift_size[0],
                               self.shift_size[1]],
                        axis=[1, 2])
        else:
            x = shifted_x

        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H * W, C))

        # Drop-path
        # if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)

        # Skip connection I (end)
        x = x_skip + x

        # Skip connection II (start)
        x_skip = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        # Skip connection II (end)
        x = x_skip + x

        return x


class ContextSwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, act="gelu", norm="layer",
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        # number of embedded patches; a tuple of  (heigh, width)
        self.num_patch = num_patch
        self.num_heads = num_heads  # number of attention heads
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)  # size of window
        else:
            self.window_size = window_size
        if isinstance(shift_size, int):
            self.shift_size = (shift_size,
                               shift_size)  # size of window shift
        else:
            self.shift_size = shift_size
        self.num_mlp = num_mlp  # number of MLP nodes
        self.prefix = name

        norm = None if swin_v2 else norm
        # Layers
        self.norm1 = get_norm_layer(norm, name='{}_norm1'.format(self.prefix))
        self.norm2 = get_norm_layer(norm, name='{}_norm2'.format(self.prefix))
        self.attn = ContextWindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=proj_drop,
                                           swin_v2=swin_v2, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm3 = get_norm_layer(norm, name='{}_norm3'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], act=act,
                       drop=mlp_drop, name=self.prefix)

        # Assertions
        for shift_size in self.shift_size:
            assert 0 <= shift_size, 'shift_size >= 0 is required'
        for shift_size, window_size in zip(self.shift_size, self.window_size):
            assert shift_size < window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < min(self.window_size):
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if np.prod(self.shift_size) > 0:
            H, W = self.num_patch
            num_window_elements = np.prod(self.window_size)
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))

            # attention mask
            mask_array = np.zeros((1, H, W, 1))

            # initialization
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows,
                                      shape=[-1, num_window_elements])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask,
                                         trainable=False,
                                         name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x, y):
        H, W = self.num_patch
        _, L, C = x.get_shape().as_list()
        num_window_elements = np.prod(self.window_size)
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)
        y = self.norm2(y)
        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))
        y = tf.reshape(y, shape=(-1, H, W, C))
        # Cyclic shift
        if np.max(self.shift_size) > 0:
            shifted_x = tf.roll(x,
                                shift=[-self.shift_size[0],
                                       -self.shift_size[1]],
                                axis=[1, 2])
            shifted_y = tf.roll(y,
                                shift=[-self.shift_size[0],
                                       -self.shift_size[1]],
                                axis=[1, 2])

        else:
            shifted_x = x
            shifted_y = y
        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows,
                               shape=(-1, num_window_elements, C))
        y_windows = window_partition(shifted_y, self.window_size)
        x_windows = tf.reshape(y_windows,
                               shape=(-1, num_window_elements, C))
        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows,
                                  shape=(-1, *self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if np.max(self.shift_size) > 0:
            x = tf.roll(shifted_x,
                        shift=[self.shift_size[0],
                               self.shift_size[1]],
                        axis=[1, 2])
        else:
            x = shifted_x

        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H * W, C))

        # Drop-path
        # if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)

        # Skip connection I (end)
        x = x_skip + x

        # Skip connection II (start)
        x_skip = x

        x = self.norm3(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        # Skip connection II (end)
        x = x_skip + x

        return x


class SwinTransformerBlock3D(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, act="gelu", norm="layer",
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, swin_v2=False, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        # number of embedded patches; a tuple of  (heigh, width)
        self.num_patch = num_patch
        self.num_heads = num_heads  # number of attention heads
        if isinstance(window_size, int):
            self.window_size = (window_size,
                                window_size,
                                window_size)  # size of window
        else:
            self.window_size = window_size
        if isinstance(shift_size, int):
            self.shift_size = (shift_size,
                               shift_size,
                               shift_size)  # size of window shift
        else:
            self.shift_size = shift_size
        self.num_mlp = num_mlp  # number of MLP nodes
        self.prefix = name
        norm = None if swin_v2 else norm

        # Layers
        self.norm1 = get_norm_layer(norm, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=proj_drop,
                                      swin_v2=swin_v2, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = get_norm_layer(norm, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], act=act,
                       drop=mlp_drop, name=self.prefix)

        # Assertions
        for shift_size in self.shift_size:
            assert 0 <= shift_size, 'shift_size >= 0 is required'
        for shift_size, window_size in zip(self.shift_size, self.window_size):
            assert shift_size < window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < min(self.window_size):
            self.shift_size = (0, 0, 0)
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if np.prod(self.shift_size) > 0:
            Z, H, W = self.num_patch
            num_window_elements = np.prod(self.window_size)
            z_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], - self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], - self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], - self.shift_size[2]),
                        slice(-self.shift_size[2], None))

            # attention mask
            mask_array = np.zeros((1, Z, H, W, 1))

            # initialization
            count = 0
            for z in z_slices:
                for h in h_slices:
                    for w in w_slices:
                        mask_array[:, z, h, w, :] = count
                        count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition_3d(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows,
                                      shape=[-1, num_window_elements])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False,
                                         name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        Z, H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        num_window_elements = np.prod(self.window_size)
        # Checking num_path and tensor sizes
        assert L == H * W * Z, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)

        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, Z, H, W, C))

        # Cyclic shift
        if np.max(self.shift_size) > 0:
            shifted_x = tf.roll(x,
                                shift=[-self.shift_size[0],
                                       -self.shift_size[1],
                                       -self.shift_size[2]],
                                axis=[1, 2, 3])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows,
                               shape=(-1, num_window_elements, C))
        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows,
                                  shape=(-1, *self.window_size, C))
        shifted_x = window_reverse_3d(attn_windows, self.window_size,
                                      Z, H, W, C)

        # Reverse cyclic shift
        if np.max(self.shift_size) > 0:
            x = tf.roll(shifted_x,
                        shift=[self.shift_size[0],
                               self.shift_size[1],
                               self.shift_size[2]],
                        axis=[1, 2, 3])
        else:
            x = shifted_x

        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H * W * Z, C))

        # Drop-path
        # if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)

        # Skip connection I (end)
        x = x_skip + x

        # Skip connection II (start)
        x_skip = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        # Skip connection II (end)
        x = x_skip + x

        return x
