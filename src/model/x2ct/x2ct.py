# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""Inception V3 model for Keras.

Reference:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""
import math
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow_addons.layers import InstanceNormalization, AdaptiveAveragePooling2D
from ..inception_resnet_v2_unet_fix.layers import get_act_layer


def get_submodules():
    return backend, layers, models, keras_utils


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normliazation="LayerNormalization",
              activation="gelu",
              name=None):
    if name is not None:
        normalization_name = name + '_in'
        conv_name = name + '_conv'
    else:
        normalization_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = InstanceNormalization(axis=bn_axis, scale=False,
                              name=normalization_name)(x)
    x = get_act_layer(activation, name)(x)
    return x


def inceptionv3(input_tensor):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules()

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(input_tensor, 32, 3, 3, strides=(2, 2),
                  padding='same', name='pooling1')
    x = conv2d_bn(x, 64, 3, 3, padding='same')
    # skip1.shape = [B 256 256 64]
    skip1 = x
    # previous: strides=(2, 2)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1),
                            padding='same', name='pooling2')(x)
    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 3, 3, padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')
    # skip2.shape = [B, 128, 128, ?]
    skip2 = x
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

    branch_pool = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')
    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
    # skip3.shape = [B, 64, 64, ?]
    skip3 = x
    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='same')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')

    branch_pool = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    # x.shape = [B, 32, 32, 32]
    # skip1.shape = [B 256, 256, 64]
    # skip2.shape = [B 128, 128, ?]
    # skip3.shape = [B 64, 64, ?]
    skip_list = [skip3, skip2, skip1]

    return x, skip_list


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


class SkipUpsample3DProgressive(layers.Layer):
    def __init__(self, filters, out_filters, z,
                 norm="instance", activation="gelu"):
        super().__init__()
        self.act_layer = get_act_layer(activation)
        self.norm_layer = get_norm_layer(norm)
        self.upsample_z_num = int(math.log(z, 2))
        for idx in range(self.upsample_z_num):

            if idx == 0:
                conv_kernel_size = (1, 3, 3)
            elif idx < self.upsample_z_num - 3:
                conv_kernel_size = 1
            elif idx < self.upsample_z_num - 1:
                conv_kernel_size = (3, 1, 1)
            elif idx == self.upsample_z_num - 1:
                conv_kernel_size = 3

            if idx in [0, self.upsample_z_num - 1]:
                upsample_block = Pixelshuffle3D(kernel_size=(2, 1, 1))
            else:
                upsample_block = layers.UpSampling3D(size=(2, 1, 1))
            conv_block = layers.Conv3D(filters, conv_kernel_size, padding="same",
                                       use_bias=False)
            setattr(self, f"upsample_{idx}", upsample_block)
            setattr(self, f"conv_{idx}", conv_block)
        self.last_conv = layers.Conv3D(out_filters, 1, padding="same",
                                       use_bias=False)

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, x):
        # x.shape = [B, ct_size, ct_size, C]
        x = x[:, None]
        # x.shape = [B, 1, ct_size, ct_size, C]
        for idx in range(self.upsample_z_num):
            x = getattr(self, f"upsample_{idx}")(x)
            x = getattr(self, f"conv_{idx}")(x)
            x = self.norm_layer(x)
            x = self.act_layer(x)
        x = self.last_conv(x)
        return x


class SkipUpsample3DTile(layers.Layer):
    def __init__(self, filters, out_filters, z,
                 norm="instance", activation="gelu"):
        super().__init__()
        self.z = z
        compress_layer_list = [
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=False),
            get_norm_layer(norm),
            get_act_layer(activation)
        ]
        self.compress_block = Sequential(compress_layer_list)
        self.conv_block = Sequential([
            layers.Conv3D(out_filters, kernel_size=3, padding="same",
                          strides=1, use_bias=False)
        ])

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, input_tensor):
        conv = self.compress_block(input_tensor)
        # shape: [B H W 1 C]
        conv = backend.expand_dims(conv, axis=1)
        # shape: [B H W Z C]
        conv = backend.repeat_elements(conv, rep=self.z, axis=1)
        conv = self.conv_block(conv)
        return conv


class SkipUpsample3DDetail(layers.Layer):
    def __init__(self, filters, out_filters, z,
                 norm="instance", activation="gelu"):
        super().__init__()
        self.block_2d_3d_progressive = SkipUpsample3DProgressive(filters, out_filters // 2,
                                                                 z, norm, activation)
        self.block_2d_3d_tile = SkipUpsample3DTile(filters, out_filters // 2,
                                                   z, norm, activation)

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, input_tensor):
        tensor_2d_3d_progressive = self.block_2d_3d_progressive(input_tensor)
        tensor_2d_3d_tile = self.block_2d_3d_tile(input_tensor)
        return backend.concatenate([tensor_2d_3d_progressive,
                                    tensor_2d_3d_tile], axis=-1)


def x2ct_model(input_shape, last_act, last_channel_num=1,
               block_2d_3d=["progressive", "tile"],
               decode_init_filters=512, decode_filter_list=[512, 256, 128, 64]):
    model_input = layers.Input(input_shape)

    encoded, skip_list = inceptionv3(model_input)
    len_decode = len(skip_list) + 1
    _, h, w, filters = backend.int_shape(encoded)

    if block_2d_3d[0] == "progressive":
        get_block_2d_3d = SkipUpsample3DProgressive
    elif block_2d_3d[0] == "tile":
        get_block_2d_3d = SkipUpsample3DTile
    elif block_2d_3d[0] == "detail":
        get_block_2d_3d = SkipUpsample3DDetail

    if block_2d_3d[1] == "progressive":
        get_skip_block_2d_3d = SkipUpsample3DProgressive
    elif block_2d_3d[1] == "tile":
        get_skip_block_2d_3d = SkipUpsample3DTile
    elif block_2d_3d[1] == "detail":
        get_skip_block_2d_3d = SkipUpsample3DDetail

    decoded = layers.Conv2D(decode_init_filters, 1,
                            padding="same")(encoded)
    decoded = get_block_2d_3d(decode_init_filters,
                              decode_init_filters, h)(decoded)

    for idx in range(len_decode):
        decode_filter = decode_filter_list[idx]
        if idx < len_decode - 1:
            skip = skip_list[idx]
            _, h, w, skip_filters = backend.int_shape(skip)

            skip = layers.Conv2D(decode_filter // 2, 1, padding="same")(skip)
            skip = get_skip_block_2d_3d(skip_filters,
                                        decode_filter // 2, h)(skip)
            decoded = layers.UpSampling3D(size=(2, 2, 2))(decoded)
            decoded = layers.Concatenate(axis=-1)([decoded, skip])
            norm = "instance"
            act = "gelu"
        else:
            decoded = Pixelshuffle3D(kernel_size=(2, 2, 2))(decoded)
            norm = None
            act = None
        decoded = layers.Conv3D(decode_filter, 3, 1, padding="same",
                                use_bias=False)(decoded)
        decoded = get_norm_layer(norm)(decoded)
        decoded = get_act_layer(act)(decoded)
    decoded = layers.Conv3D(last_channel_num, 3, 1, padding="same",
                            use_bias=False)(decoded)
    decoded = get_act_layer(last_act)(decoded)

    return Model(model_input, decoded)


def disc_model(input_shape, last_act):
    model_input = layers.Input(input_shape)

    encoded, skip_list = inceptionv3(model_input)

    _, h, w, encoded_filter = backend.int_shape(encoded)
    len_skip = len(skip_list)
    skip_list = skip_list[::-1]
    for idx, skip in enumerate(skip_list):
        encoded_filter //= 2
        if idx < len_skip - 1:
            norm = "instance"
            act = "gelu"
        else:
            norm = None
            act = None
        _, _, _, skip_filter = backend.int_shape(skip)
        skip = AdaptiveAveragePooling2D((h, w))(skip)
        encoded = layers.Conv2D(skip_filter,
                                1, use_bias=False)(encoded)
        encoded = layers.Add()([encoded, skip])
        encoded = layers.Conv2D(encoded_filter, 3, 1, padding="same",
                                use_bias=False)(encoded)
        encoded = get_norm_layer(norm)(encoded)
        encoded = get_act_layer(act)(encoded)

    encoded_filter //= 2
    encoded = layers.Conv2D(encoded_filter, 3, 1, use_bias=False)(encoded)
    encoded = layers.Conv2D(1, 1, 1, use_bias=False)(encoded)
    encoded = get_act_layer(last_act)(encoded)

    return Model(model_input, encoded)


def disc_model(input_shape, last_act):
    model_input = layers.Input(input_shape)

    encoded, skip_list = inceptionv3(model_input)

    _, h, w, encoded_filter = backend.int_shape(encoded)
    len_skip = len(skip_list)
    skip_list = skip_list[::-1]
    for idx, skip in enumerate(skip_list):
        encoded_filter //= 2
        if idx < len_skip - 1:
            norm = "instance"
            act = "gelu"
        else:
            norm = None
            act = None
        _, _, _, skip_filter = backend.int_shape(skip)
        skip = AdaptiveAveragePooling2D((h, w))(skip)
        encoded = layers.Conv2D(skip_filter,
                                1, use_bias=False)(encoded)
        encoded = layers.Add()([encoded, skip])
        encoded = layers.Conv2D(skip_filter, 3, 1, padding="same",
                                use_bias=False)(encoded)
        encoded = get_norm_layer(norm)(encoded)
        encoded = get_act_layer(act)(encoded)

    encoded_filter //= 2
    encoded = layers.Conv2D(skip_filter, 3, 1, use_bias=False)(encoded)
    encoded = layers.Conv2D(1, 1, 1, use_bias=False)(encoded)
    encoded = get_act_layer(last_act)(encoded)

    return Model(model_input, encoded)
