# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Inception-ResNet V2 model for Keras.

Reference:
  - [Inception-v4, Inception-ResNet and the Impact of
     Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
    (AAAI 2017)
"""

from tensorflow.python.keras import backend

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from .layers import TransformerEncoder2D
from .transformer_layers import AddPositionEmbs
from .cbam_attention_module import attach_attention_module

SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "mixed_5b", "mixed_6a", "mixed_7a"]
CHANNEL_AXIS = -1


def downsize_hw(h, w):
    return h // 2, w // 2


def get_act_layer(activation, name=None):
    if activation is None:
        act_layer = layers.Lambda(lambda x: x, name=name)
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6, name=name)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.3, name=name)
    else:
        act_layer = layers.Activation(activation, name=name)
    return act_layer


def InceptionResNetV2(input_shape=None,
                      block_size=16,
                      padding="valid",
                      base_act="relu",
                      last_act="relu",
                      name_prefix="",
                      use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    init_conv = get_init_conv(input_shape, block_size, base_act,
                              5, name_prefix)
    block_1 = get_block_1(init_conv.input.shape[1:], block_size,
                          padding, base_act, name_prefix)
    block_2 = get_block_2(block_1.input.shape[1:], block_size,
                          padding, base_act, name_prefix)
    block_3 = get_block_3(block_2.input.shape[1:], block_size,
                          padding, base_act, name_prefix)
    block_4 = get_block_4(block_3.input.shape[1:], block_size,
                          use_attention, padding, base_act, name_prefix)
    block_5 = get_block_5(block_4.input.shape[1:], block_size,
                          use_attention, padding, base_act, name_prefix)
    output_block = get_output_block(block_5.input.shape[1:], block_size,
                                    use_attention, base_act, last_act, name_prefix)

    input_tensor = layers.Input(input_shape)
    x = init_conv(input_tensor)
    x = block_1(x)
    x = block_2(x)
    x = block_3(x)
    x = block_4(x)
    x = block_5(x)
    output_block_tensor = output_block(x)

    return Model(input_tensor, output_block_tensor,
                 name=f'{name_prefix}inception_resnet_v2')


def InceptionResNetV2_progressive(target_shape=None,
                                  block_size=16,
                                  padding="same",
                                  base_act="relu",
                                  last_act="relu",
                                  name_prefix="",
                                  num_downsample=5,
                                  use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    final_downsample = 5

    input_shape = (target_shape[0] // (2 ** (final_downsample - num_downsample)),
                   target_shape[1] // (2 **
                                       (final_downsample - num_downsample)),
                   target_shape[2])
    H, W, _ = input_shape

    input_tensor = layers.Input(input_shape)
    x = get_init_conv(input_tensor, block_size, base_act,
                      num_downsample, name_prefix)
    if num_downsample >= 5:
        x = get_block_1(x, block_size,
                        padding, base_act, name_prefix)
    if num_downsample >= 4:
        x = get_block_2(x, block_size,
                        padding, base_act, name_prefix)
    if num_downsample >= 3:
        x = get_block_3(x, block_size,
                        padding, base_act, name_prefix)
    if num_downsample >= 2:
        x = get_block_4(x, block_size, padding,
                        use_attention, base_act, name_prefix)
    if num_downsample >= 1:
        x = get_block_5(x, block_size, padding,
                        use_attention, base_act, name_prefix)
    x = get_output_block(x, block_size,
                         use_attention, base_act, last_act, name_prefix)

    return Model(input_tensor, x,
                 name=f'{name_prefix}inception_resnet_v2')


def get_init_conv(input_tensor, block_size,
                  activation, num_downsample, name_prefix):
    if num_downsample == 5:
        output_filter = block_size * 2
    elif num_downsample == 4:
        output_filter = block_size * 2
    elif num_downsample == 3:
        output_filter = block_size * 4
    elif num_downsample == 2:
        output_filter = block_size * 12
    elif num_downsample == 1:
        output_filter = block_size * 68
    conv = Conv2DBN(output_filter, 1, padding="same",
                    activation=activation,
                    name=f"{name_prefix}init_conv")(input_tensor)
    return conv


def get_block_1(input_tensor, block_size,
                padding, activation, name_prefix):
    conv = Conv2DBN(block_size * 2, 3, strides=2, padding=padding,
                    activation=activation,
                    name=f"{name_prefix}down_block_1")(input_tensor)
    return conv


def get_block_2(input_tensor, block_size,
                padding, activation, name_prefix):
    conv_1 = Conv2DBN(block_size * 2, 3, padding=padding,
                      activation=activation)(input_tensor)
    conv_2 = Conv2DBN(block_size * 4, 3,
                      activation=activation)(conv_1)
    max_pool = layers.MaxPooling2D(3, strides=2, padding=padding,
                                   name=f"{name_prefix}down_block_2")(conv_2)
    return max_pool


def get_block_3(input_tensor, block_size,
                padding, activation, name_prefix):
    conv_1 = Conv2DBN(block_size * 5, 1, padding=padding,
                      activation=activation)(input_tensor)
    conv_2 = Conv2DBN(block_size * 12, 3, padding=padding,
                      activation=activation)(conv_1)
    max_pool = layers.MaxPooling2D(3, strides=2, padding=padding,
                                   name=f"{name_prefix}down_block_3")(conv_2)
    return max_pool


def get_block_4(input_tensor, block_size, padding,
                use_attention, activation, name_prefix):
    branch_0 = Conv2DBN(block_size * 6, 1,
                        activation=activation)(input_tensor)
    branch_1 = Conv2DBN(block_size * 3, 1,
                        activation=activation)(input_tensor)
    branch_1 = Conv2DBN(block_size * 4, 5,
                        activation=activation)(branch_1)
    branch_2 = Conv2DBN(block_size * 4, 1,
                        activation=activation)(input_tensor)
    branch_2 = Conv2DBN(block_size * 6, 3,
                        activation=activation)(branch_2)
    branch_2 = Conv2DBN(block_size * 6, 3,
                        activation=activation)(branch_2)
    branch_pool = layers.AveragePooling2D(
        3, strides=1, padding='same')(input_tensor)
    branch_pool = Conv2DBN(block_size * 4, 1,
                           activation=activation)(branch_pool)
    branches_1 = [branch_0, branch_1, branch_2, branch_pool]

    branches_1 = layers.Concatenate(axis=CHANNEL_AXIS)(branches_1)

    for idx in range(1, 11):
        branches_1 = InceptionResnetBlock(scale=0.17,
                                          block_type='block35', block_size=block_size,
                                          activation=activation,
                                          use_attention=use_attention,
                                          name=f'block_35_{idx}')(branches_1)
    branch_0 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        activation=activation)(branches_1)
    branch_1 = Conv2DBN(block_size * 16, 1,
                        activation=activation)(branches_1)
    branch_1 = Conv2DBN(block_size * 16, 3,
                        activation=activation)(branch_1)
    branch_1 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        activation=activation)(branch_1)
    branch_pool = layers.AveragePooling2D(
        3, strides=2, padding=padding)(branches_1)
    branches_2 = [branch_0, branch_1, branch_pool]
    branches_2 = layers.Concatenate(axis=CHANNEL_AXIS,
                                    name=f"{name_prefix}down_block_4")(branches_2)

    return branches_2


def get_block_5(input_tensor, block_size, padding,
                use_attention, activation, name_prefix):
    branches_1 = input_tensor
    for idx in range(1, 21):
        branches_1 = InceptionResnetBlock(scale=0.11,
                                          block_type='block17', block_size=block_size,
                                          activation=activation,
                                          use_attention=use_attention,
                                          name=f'block_17_{idx}')(branches_1)
    branch_0 = Conv2DBN(block_size * 16, 1,
                        activation=activation)(branches_1)
    branch_0 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        activation=activation)(branch_0)
    branch_1 = Conv2DBN(block_size * 16, 1,
                        activation=activation)(branches_1)
    branch_1 = Conv2DBN(block_size * 18, 3, strides=2, padding=padding,
                        activation=activation)(branch_1)
    branch_2 = Conv2DBN(block_size * 16, 1,
                        activation=activation)(branches_1)
    branch_2 = Conv2DBN(block_size * 18, 3,
                        activation=activation)(branch_2)
    branch_2 = Conv2DBN(block_size * 20, 3, strides=2, padding=padding,
                        activation=activation)(branch_2)
    branch_pool = layers.MaxPooling2D(3, strides=2,
                                      padding=padding)(branches_1)
    branches_2 = [branch_0, branch_1, branch_2, branch_pool]
    branches_2 = layers.Concatenate(axis=CHANNEL_AXIS,
                                    name=f"{name_prefix}down_block_5")(branches_2)
    return branches_2


def get_output_block(input_tensor, block_size, use_attention,
                     activation, last_activation, name_prefix):
    branches = input_tensor
    for idx in range(1, 11):
        branches = InceptionResnetBlock(scale=0.2,
                                        block_type='block8', block_size=block_size,
                                        activation=activation,
                                        use_attention=use_attention,
                                        name=f'block_8_{idx}')(branches)
    branches = Conv2DBN(block_size * 96, 1,
                        activation=last_activation,
                        name=f"{name_prefix}output_block")(branches)
    return branches


class Conv2DBN(layers.Layer):
    def __init__(self, filters, kernel_size,
                 strides=1, padding="same",
                 activation="relu", use_bias=False, name=None):
        super().__init__()
        bn_axis = CHANNEL_AXIS
        self.conv_layer = layers.Conv2D(filters, kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        use_bias=use_bias)
        self.norm_layer = layers.BatchNormalization(axis=bn_axis,
                                                    scale=False)
        self.act_layer = get_act_layer(activation, name=name)

    def __call__(self, input_tensor):
        conv = self.conv_layer(input_tensor)
        norm = self.norm_layer(conv)
        act = self.act_layer(norm)
        return act


class ConcatBlock(layers.Layer):
    def __init__(self, layer_list):
        super().__init__()
        concat_axis = CHANNEL_AXIS
        self.layer_list = layer_list
        self.concat_layer = layers.Concatenate(axis=concat_axis)

    def call(self, input_tensor):
        tensor_list = [layer(input_tensor) for layer in self.layer_list]
        concat = self.concat_layer(tensor_list)
        return concat


class InceptionResnetBlock(layers.Layer):
    def __init__(self, scale, block_type,
                 block_size=16, activation='relu',
                 use_attention=True, name=None):
        super().__init__(name=name)
        self.use_attention = use_attention
        if block_type == 'block35':
            branch_0 = Conv2DBN(block_size * 2, 1,
                                activation=activation)
            branch_1_1 = Conv2DBN(block_size * 2, 1, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 2, 3, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2])
            branch_2_1 = Conv2DBN(block_size * 2, 1, activation=activation)
            branch_2_2 = Conv2DBN(block_size * 3, 3, activation=activation)
            branch_2_3 = Conv2DBN(block_size * 4, 3, activation=activation)
            branch_2 = Sequential([branch_2_1,
                                   branch_2_2,
                                   branch_2_3])
            branches = [branch_0, branch_1, branch_2]
            up_channel = block_size * 20
        elif block_type == 'block17':
            branch_0 = Conv2DBN(block_size * 12, 1,
                                activation=activation)
            branch_1_1 = Conv2DBN(block_size * 8, 1, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 10, [1, 7],
                                  activation=activation)
            branch_1_3 = Conv2DBN(block_size * 10, [7, 1],
                                  activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3])
            branches = [branch_0, branch_1]
            up_channel = block_size * 68
        elif block_type == 'block8':
            branch_0 = Conv2DBN(block_size * 12, 1,
                                activation=activation)
            branch_1_1 = Conv2DBN(block_size * 12, 1, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 14, [1, 3],
                                  activation=activation)
            branch_1_3 = Conv2DBN(block_size * 16, [3, 1],
                                  activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3])
            branches = [branch_0, branch_1]
            up_channel = block_size * 130
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.branch_layer = ConcatBlock(branches)
        self.up_layer = Conv2DBN(up_channel, 1,
                                 activation=None, use_bias=True)
        if self.use_attention:
            self.attention_layer = CBAM_Block2D(up_channel, ratio=8)
        self.residual_block = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=up_channel,
            arguments={'scale': scale})
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):
        branch = self.branch_layer(input_tensor)
        up = self.up_layer(branch)
        if self.use_attention:
            up = self.attention_layer(up)
        residual = self.residual_block([input_tensor, up])
        act = self.act_layer(residual)
        return act


class CBAM_Block2D(layers.Layer):
    def __init__(self, input_filter, ratio=8):
        super().__init__()
        self.channel_attention_layer = ChannelAttention2D(input_filter,
                                                          ratio=ratio)
        self.spatial_attention_layer = SpatialAttention2D(input_filter)

    def call(self, input_tensor):
        channel_attention = self.channel_attention_layer(input_tensor)
        spatial_attention = self.spatial_attention_layer(channel_attention)
        return spatial_attention


class ChannelAttention2D(layers.Layer):
    def __init__(self, input_filter, ratio=8):
        super().__init__()
        self.shared_dense_one = layers.Dense(input_filter // ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_dense_two = layers.Dense(input_filter,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.avg_pool_layer = layers.GlobalAveragePooling2D()
        self.max_pool_layer = layers.GlobalMaxPooling2D()
        self.reshape_layer = layers.Reshape((1, 1, input_filter))
        self.act_layer = get_act_layer('sigmoid')

    def call(self, input_tensor):

        avg_pool = self.avg_pool_layer(input_tensor)
        avg_pool = self.reshape_layer(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.max_pool_layer(input_tensor)
        max_pool = self.reshape_layer(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        cbam_feature = avg_pool + max_pool
        cbam_feature = self.act_layer(cbam_feature)
        output = input_tensor * cbam_feature
        return output


class SpatialAttention2D(layers.Layer):
    def __init__(self, input_filter):
        super().__init__()
        kernel_size = 7
        self.avg_pool_layer = layers.Lambda(
            lambda x: backend.mean(x, axis=-1, keepdims=True))
        self.max_pool_layer = layers.Lambda(
            lambda x: backend.max(x, axis=-1, keepdims=True))
        self.concat_layer = layers.Concatenate(axis=-1)
        self.cbam_conv_layer = layers.Conv2D(filters=1,
                                             kernel_size=kernel_size,
                                             strides=1,
                                             padding='same',
                                             activation='sigmoid',
                                             kernel_initializer='he_normal',
                                             use_bias=False)

    def call(self, input_tensor):
        avg_pool = self.avg_pool_layer(input_tensor)
        max_pool = self.max_pool_layer(input_tensor)
        concat = self.concat_layer([avg_pool, max_pool])
        cbam_feature = self.cbam_conv_layer(concat)
        output = input_tensor * cbam_feature
        return output
