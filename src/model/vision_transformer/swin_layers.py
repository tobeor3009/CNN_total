
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.activations import softmax

from .util_layers import drop_path, get_act_layer


def meshgrid_3d(*arrs):
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(len, arrs))
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
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size,
                   patch_num_W, window_size, C))
    # (-1, patch_num_H, patch_num_W, window_size, window_size, C)
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))

    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))

    return windows


def window_partition_3d(x, window_size):

    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, Z, H, W, C = x.get_shape().as_list()
    # Subset tensors to patches
    patch_num_Z = Z // window_size
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(x, shape=(-1, patch_num_Z, window_size,
                   patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))

    # Reshape patches to a patch sequence
    windows = tf.reshape(x,
                         shape=(-1, window_size, window_size, window_size, C))

    return windows


def window_reverse(windows, window_size, H, W, C):

    # Reshape a patch sequence to aligned patched
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W,
                                   window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))

    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))

    return x


def window_reverse_3d(windows, window_size, Z, H, W, C):

    # Reshape a patch sequence to aligned patched
    patch_num_Z = Z // window_size
    patch_num_H = H // window_size
    patch_num_W = W // window_size
    x = tf.reshape(windows, shape=(-1, patch_num_Z, patch_num_H, patch_num_W,
                                   window_size, window_size, window_size, C))
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
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.window_size = window_size  # size of the attention window
        self.num_heads = num_heads  # number of self-attention heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # query scaling factor

        self.prefix = name

        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):

        # zero initialization
        num_window_elements = (
            2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                            shape=(
                                                                num_window_elements, self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
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
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        # Query rescaling
        q = q * self.scale

        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)

        # Shift window
        num_window_elements = np.prod(self.window_size)
        relative_position_index_flat = tf.reshape(self.relative_position_index,
                                                  shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table,
                                           relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias,
                                              perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

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


class WindowAttention3D(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        self.window_size = window_size  # size of the attention window
        self.num_heads = num_heads  # number of self-attention heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # query scaling factor

        self.prefix = name

        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):

        # zero initialization
        num_window_elements = np.prod((
            2 * self.window_size[0] - 1,
            2 * self.window_size[1] - 1,
            2 * self.window_size[2] - 1
        ))
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
        coords_flatten = coords.reshape(3, -1)
        # relative_coords.shape = 3, 8
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
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
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        # Query rescaling
        q = q * self.scale

        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))

        attn = (q @ k)
        # Shift window
        num_window_elements = np.prod(self.window_size)
        relative_position_index_flat = tf.reshape(self.relative_position_index,
                                                  shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table,
                                           relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=(num_window_elements, num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias,
                                              perm=(3, 0, 1, 2))
        attn = attn + relative_position_bias

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


class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, act="gelu",
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        # number of embedded patches; a tuple of  (heigh, width)
        self.num_patch = num_patch
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        self.prefix = name

        # Layers
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], act=act,
                       drop=mlp_drop, name=self.prefix)

        # Assertions
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -
                        self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -
                        self.shift_size), slice(-self.shift_size, None))

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
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)

        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows,
                                  shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
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


class SwinTransformerBlock3D(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, act="gelu",
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, name=''):
        super().__init__()

        self.dim = dim  # number of input dimensions
        # number of embedded patches; a tuple of  (heigh, width)
        self.num_patch = num_patch
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes
        self.prefix = name

        # Layers
        self.norm1 = LayerNormalization(
            epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention3D(dim, window_size=(self.window_size, self.window_size, self.window_size), num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(
            epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], act=act,
                       drop=mlp_drop, name=self.prefix)

        # Assertions
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size > 0:
            Z, H, W = self.num_patch
            z_slices = (slice(0, -self.window_size), slice(-self.window_size, -
                        self.shift_size), slice(-self.shift_size, None))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -
                        self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -
                        self.shift_size), slice(-self.shift_size, None))

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
                                      shape=[-1, self.window_size ** 3])
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

        # Checking num_path and tensor sizes
        assert L == H * W * Z, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)

        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, Z, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x,
                                shift=[-self.shift_size,
                                       -self.shift_size,
                                       -self.shift_size],
                                axis=[1, 2, 3])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows,
                               shape=(-1, self.window_size ** 3, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows,
                                  shape=(-1, self.window_size, self.window_size, self.window_size, C))
        shifted_x = window_reverse_3d(attn_windows, self.window_size,
                                      Z, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x,
                        shift=[self.shift_size,
                               self.shift_size,
                               self.shift_size],
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
