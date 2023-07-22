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
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow_addons.layers import InstanceNormalization
from ..inception_resnet_v2_unet_fix.layers import get_act_layer
from .util import BASE_ACT, RGB_OUTPUT_CHANNEL, SEG_OUTPUT_CHANNEL


def classification_block(x, skip_list, base_act, last_act, class_num, grad_cam):
    x = layers.GlobalAveragePooling2D()(x)

    for skip in skip_list:
        skip_channel = skip.shape[-1]
        skip = layers.GlobalAveragePooling2D()(skip)
        skip = layers.Dense(skip_channel // 2)(skip)
        skip = get_act_layer(base_act)(skip)
        x = layers.Concatenate(axis=-1)([x, skip])
    channel = x.shape.as_list()[-1]
    x = layers.Dense(channel // 2)(x)
    x = get_act_layer(base_act)(x)
    if grad_cam:
        # x *= 1e-1
        backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"
    x = layers.Dense(class_num, dtype=dense_dtype)(x)
    x = get_act_layer(last_act, name="classification_output")(x)
    return x


def pixel_shuffle_block(x, skip_list, base_act, last_act, class_num):

    # skip1.shape = [B 256, 256, 32]
    # skip2.shape = [B 128, 128, 64]
    # skip3.shape = [B 64, 64, 192]
    # skip4.shape = [B 32, 32, 768]
    skip1, skip2, skip3, skip4 = skip_list

    def pixel_shuffle_mini_block(x, block_size, skip, idx):
        skip_channel_list = [64, 48, 36, 24, 12]
        if idx > 1:
            x = layers.Conv2D(block_size * 4, (1, 1), padding='same',
                              name=f'decoder_pointwise_{idx}')(x)
            x = InstanceNormalization(name=f'decoder_pointwise_IN_{idx}',
                                      epsilon=1e-3)(x)
        x = tf.nn.depth_to_space(x, block_size=2)
        if skip is not None:
            dec_skip = conv2d_bn(skip, skip_channel_list[idx - 1], 1, 1,
                                 padding="same", strides=(1, 1), activation=base_act,
                                 name=f'recon_feature_projection_{idx}')
            x = layers.Concatenate()([x, dec_skip])
        x = conv2d_bn(x, block_size, 3, 3,
                      padding="same", strides=(1, 1), activation=base_act,
                      name=f'recon_feature_conv_{idx}')
        return x
    # x.shape = [B, 16, 16, 1596]
    x = pixel_shuffle_mini_block(x, 384, skip4, 1)
    # x.shape = [B, 32, 32, 32]
    x = pixel_shuffle_mini_block(x, 192, skip3, 2)
    # x.shape = [B, 64, 64, 192]
    x = pixel_shuffle_mini_block(x, 128, skip2, 3)
    # x.shape = [B, 128, 128, 128]
    x = pixel_shuffle_mini_block(x, 96, skip1, 4)
    # x.shape = [B, 256, 256, 96]
    x = pixel_shuffle_mini_block(x, 64, None, 5)
    # x.shape = [B, 512, 512, 64]
    x = layers.Conv2D(class_num, (1, 1),
                      padding='same')(x)
    # img_input.shape = [B, 512, 512, 3]
    x = get_act_layer(last_act, name="pixel_shuffle_output")(x)
    return x


def upsample_block(x, skip_list, base_act, last_act, class_num):

    # skip1.shape = [B 256, 256, 32]
    # skip2.shape = [B 128, 128, 64]
    # skip3.shape = [B 64, 64, 192]
    # skip4.shape = [B 32, 32, 768]
    skip1, skip2, skip3, skip4 = skip_list

    def upsample_mini_block(x, block_size, skip, idx):
        skip_channel_list = [64, 48, 36, 24, 12]
        x = layers.UpSampling2D(size=(2, 2))(x)
        if skip is not None:
            dec_skip = conv2d_bn(skip, skip_channel_list[idx - 1], 1, 1,
                                 padding="same", strides=(1, 1),
                                 activation=base_act,
                                 name=f'seg_feature_projection_{idx}')
            x = layers.Concatenate()([x, dec_skip])
        x = conv2d_bn(x, block_size, 3, 3,
                      padding="same", strides=(1, 1), activation=base_act,
                      name=f'seg_feature_conv_{idx}')
        return x
    # x.shape = [B, 16, 16, 1596]
    x = upsample_mini_block(x, 384, skip4, 1)
    # x.shape = [B, 32, 32, 32]
    x = upsample_mini_block(x, 192, skip3, 2)
    # x.shape = [B, 64, 64, 192]
    x = upsample_mini_block(x, 128, skip2, 3)
    # x.shape = [B, 128, 128, 128]
    x = upsample_mini_block(x, 96, skip1, 4)
    # x.shape = [B, 256, 256, 96]
    x = upsample_mini_block(x, 64, None, 5)
    # x.shape = [B, 512, 512, 64]
    x = layers.Conv2D(class_num, (1, 1),
                      padding='same')(x)
    # img_input.shape = [B, 512, 512, 3]
    x = get_act_layer(last_act, name="upsample_output")(x)
    return x


def consistency_block(class_pred, seg_pred):
    seg_pred = layers.MaxPooling2D(pool_size=(32, 32),
                                   strides=(32, 32),
                                   padding="valid")(seg_pred)
    seg_pred = layers.AveragePooling2D(pool_size=(16, 16),
                                       strides=(16, 16),
                                       padding="valid")(seg_pred)
    seg_pred = seg_pred[:, 0, 0]
    consistency = tf.keras.layers.Subtract(name="consistency_output")([class_pred,
                                                                       seg_pred])
    return consistency


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
                recon_class_num=3, seg_class_num=1, classification_class_num=1,
                base_act=BASE_ACT, image_last_act="tanh", last_act="sigmoid",
                multi_task=False, classification=False, check_consistency=False,
                grad_cam=False):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules()

    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2),
                  padding='same', name='pooling1')
    x = conv2d_bn(x, 64, 3, 3, padding='same')
    skip1 = x
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same', name='pooling2')(x)
    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 3, 3, padding='same')
    skip2 = x
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
    skip3 = x
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
    skip4 = x
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
    # skip1.shape = [B 256, 256, 64]
    # skip2.shape = [B 128, 128, 192]
    # skip3.shape = [B 64, 64, 288]
    # skip4.shape = [B 32, 32, 768]
    skip_list = [skip1, skip2, skip3, skip4]

    seg_output = upsample_block(x, skip_list, base_act, last_act,
                                seg_class_num)

    if multi_task:
        he_output = pixel_shuffle_block(x, skip_list, base_act, image_last_act,
                                        recon_class_num)
        model_output = [he_output, seg_output]
    else:
        model_output = [seg_output]

    if classification:
        classification_output = classification_block(x, skip_list, base_act, "softmax",
                                                     classification_class_num, grad_cam)
        model_output.append(classification_output)

    if check_consistency:
        consistency = consistency_block(classification_output, seg_output)
        model_output.append(consistency)
    return Model(img_input, model_output,
                 name='inceptionv3')
