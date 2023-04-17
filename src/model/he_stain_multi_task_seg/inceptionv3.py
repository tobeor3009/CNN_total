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
import os

from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils as keras_utils
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.util.tf_export import keras_export
from ..inception_resnet_v2_unet_fix.layers import get_act_layer
from .util import BASE_ACT, RGB_OUTPUT_CHANNEL, SEG_OUTPUT_CHANNEL


def pixel_shuffle_block(x, skip_list, base_act, last_act):

    # skip1.shape = [B 512, 512, 32]
    # skip2.shape = [B 256, 256, 32]
    # skip3.shape = [B 128, 128, 64]
    # skip4.shape = [B 64, 64, 192]
    # skip5.shape = [B 32, 32, 768]
    skip1, skip2, skip3, skip4, skip5 = skip_list

    def pixel_shuffle_mini_block(x, block_size, skip, idx):
        skip_channel_list = [64, 48, 36, 24, 12]
        if idx > 1:
            x = layers.Conv2D(block_size * 4, (1, 1), padding='same',
                              name=f'decoder_pointwise_{idx}')(x)
            x = InstanceNormalization(name=f'decoder_pointwise_IN_{idx}',
                                      epsilon=1e-3)(x)
        x = tf.nn.depth_to_space(x, block_size=(2, 2))
        dec_skip = conv2d_bn(skip, skip_channel_list[idx], (1, 1), (1, 1),
                             padding="same", activation=base_act,
                             name=f'feature_projection_{idx}')
        x = layers.Concatenate()([x, dec_skip])
        x = conv2d_bn(x, block_size, (3, 3), (1, 1),
                      padding="same", activation=base_act,
                      name=f'feature_conv_{idx}')
    # x.shape = [B, 16, 16, 1596]
    x = pixel_shuffle_mini_block(x, 384, skip5, 1)
    # x.shape = [B, 32, 32, 32]
    x = pixel_shuffle_mini_block(x, 192, skip4, 2)
    # x.shape = [B, 64, 64, 192]
    x = pixel_shuffle_mini_block(x, 128, skip3, 3)
    # x.shape = [B, 128, 128, 128]
    x = pixel_shuffle_mini_block(x, 96, skip2, 4)
    # x.shape = [B, 256, 256, 96]
    x = pixel_shuffle_mini_block(x, 64, skip1, 5)
    # x.shape = [B, 512, 512, 64]
    x = layers.Conv2D(RGB_OUTPUT_CHANNEL, (1, 1),
                      padding='same')(x)
    # img_input.shape = [B, 512, 512, 3]
    x = get_act_layer(last_act)(x)
    return x


def upsample_block(x, skip_list, base_act, last_act):

    # skip1.shape = [B 512, 512, 32]
    # skip2.shape = [B 256, 256, 32]
    # skip3.shape = [B 128, 128, 64]
    # skip4.shape = [B 64, 64, 192]
    # skip5.shape = [B 32, 32, 768]
    skip1, skip2, skip3, skip4, skip5 = skip_list

    def upsample_mini_block(x, block_size, skip, idx):
        skip_channel_list = [64, 48, 36, 24, 12]
        x = layers.UpSampling2D(size=(2, 2))
        dec_skip = conv2d_bn(skip, skip_channel_list[idx], (1, 1), (1, 1),
                             padding="same", activation=base_act,
                             name=f'feature_projection_{idx}')
        x = layers.Concatenate()([x, dec_skip])
        x = conv2d_bn(x, block_size, (3, 3), (1, 1),
                      padding="same", activation=base_act,
                      name=f'feature_conv_{idx}')
    # x.shape = [B, 16, 16, 1596]
    x = upsample_mini_block(x, 384, skip5, 1)
    # x.shape = [B, 32, 32, 32]
    x = upsample_mini_block(x, 192, skip4, 2)
    # x.shape = [B, 64, 64, 192]
    x = upsample_mini_block(x, 128, skip3, 3)
    # x.shape = [B, 128, 128, 128]
    x = upsample_mini_block(x, 96, skip2, 4)
    # x.shape = [B, 256, 256, 96]
    x = upsample_mini_block(x, 64, skip1, 5)
    # x.shape = [B, 512, 512, 64]
    x = layers.Conv2D(SEG_OUTPUT_CHANNEL, (1, 1),
                      padding='same')(x)
    # img_input.shape = [B, 512, 512, 3]
    x = get_act_layer(last_act)(x)
    return x


def get_submodules():
    return backend, layers, models, keras_utils


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normliazation="LayerNormalization",
              activation=BASE_ACT,
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


def InceptionV3(input_shape=(512, 512, 3),
                base_act="gelu", image_last_act="tanh", last_act="sigmoid", multi_task=False):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules()

    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, padding='same')
    skip1 = x
    x = conv2d_bn(x, 32, 3, 3, strides=(2, 2),
                  padding='same', name='pooling1')
    skip2 = x
    x = conv2d_bn(x, 64, 3, 3, padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same', name='pooling2')(x)
    skip3 = x
    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 3, 3, padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    skip4 = x
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
    skip5 = x
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
    # x.shape = [B, 16, 16, 32]
    # skip1.shape = [B 512, 512, 32]
    # skip2.shape = [B 256, 256, 32]
    # skip3.shape = [B 128, 128, 64]
    # skip4.shape = [B 64, 64, 192]
    # skip5.shape = [B 32, 32, 768]
    skip_list = [skip1, skip2, skip3, skip4, skip5]

    seg_output = upsample_block(x, skip_list, base_act, last_act)
    if multi_task:
        h_output = pixel_shuffle_block(x, skip_list, base_act, image_last_act)
        e_output = pixel_shuffle_block(x, skip_list, base_act, image_last_act)
        model = Model(img_input, [h_output, e_output,
                      seg_output], name='inceptionv3')
    else:
        model = Model(img_input, seg_output, name='inceptionv3')

    return model
