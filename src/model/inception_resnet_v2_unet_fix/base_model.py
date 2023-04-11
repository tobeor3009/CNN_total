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
from tensorflow.keras import layers, Model
from .layers import TransformerEncoder2D
from .layers import get_norm_layer
from .transformer_layers import AddPositionEmbs
from .cbam_attention_module import attach_attention_module

SKIP_CONNECTION_LAYER_NAMES = ["conv_down_1_ac",
                               "maxpool_1", "mixed_5b", "mixed_6a", "mixed_7a"]


# level_1, channel(32): conv_down_1_ac
# level_2, channel(64): maxpool_1
# level_3, channel(192): maxpool_2
# level_4, channel(1088): mixed_6a
# level_5, channel(2080): mixed_7a
def InceptionResNetV2(input_shape=None,
                      block_size=16,
                      padding="valid",
                      base_act="relu",
                      last_act="relu",
                      name_prefix="",
                      attention_module="cbam_block",
                      **kwargs):

    # Determine proper input shape
    # min_size default:75
    img_input = layers.Input(shape=input_shape)

    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    # input shape: [B 512 512 3]
    # Stem block: [B 128 128 192]
    # x.shape: [B 256 256 32]
    x = conv2d_bn(img_input, block_size * 2, 3, strides=2, padding=padding,
                  activation=base_act,
                  name=f"{name_prefix}conv_down_1")
    x = conv2d_bn(x, block_size * 2, 3,
                  padding=padding, activation=base_act)
    x = conv2d_bn(x, block_size * 4, 3, activation=base_act)
    # x.shape: [B 128 128 64]
    x = layers.MaxPooling2D(3, strides=2, padding=padding,
                            name=f"{name_prefix}maxpool_1")(x)
    x = conv2d_bn(x, block_size * 5, 1, padding=padding, activation=base_act)
    x = conv2d_bn(x, block_size * 12, 3, padding=padding, activation=base_act)
    # x.shape: [B 64 64 192]
    x = layers.MaxPooling2D(3, strides=2, padding=padding,
                            name=f"{name_prefix}maxpool_2")(x)

    # Mixed 5b (Inception-A block): [B 64 64 320] or 35 x 35 x 320
    branch_0 = conv2d_bn(x, block_size * 6, 1, activation=base_act)
    branch_1 = conv2d_bn(x, block_size * 3, 1, activation=base_act)
    branch_1 = conv2d_bn(branch_1, block_size * 4, 5, activation=base_act)
    branch_2 = conv2d_bn(x, block_size * 4, 1, activation=base_act)
    branch_2 = conv2d_bn(branch_2, block_size * 6, 3, activation=base_act)
    branch_2 = conv2d_bn(branch_2, block_size * 6, 3, activation=base_act)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            4, 1, activation=base_act)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(
        axis=channel_axis, name=f'{name_prefix}mixed_5b')(branches)
    # 10x block35 (Inception-ResNet-A block): [B 64 64 320] or 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17,
                                   activation=base_act,
                                   block_type='block35', block_idx=block_idx,
                                   attention_module=attention_module,
                                   name_prefix=name_prefix)

    # Mixed 6a (Reduction-A block): [B 32 32 1088] or 17 x 17 x 1088
    branch_0 = conv2d_bn(x, block_size * 24, 3, strides=2,
                         padding=padding, activation=base_act)
    branch_1 = conv2d_bn(x, block_size * 16, 1, activation=base_act)
    branch_1 = conv2d_bn(branch_1, block_size * 16, 3, activation=base_act)
    branch_1 = conv2d_bn(branch_1, block_size * 24, 3,
                         strides=2, padding=padding, activation=base_act)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding=padding)(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(
        axis=channel_axis, name=f'{name_prefix}mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): [B 32 32 1088] or 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1,
                                   activation=base_act,
                                   block_type='block17', block_idx=block_idx,
                                   attention_module=attention_module,
                                   name_prefix=name_prefix)
    # Mixed 7a (Reduction-B block): [B 16 16 2080] or 8 x 8 x 2080
    branch_0 = conv2d_bn(x, block_size * 16, 1, activation=base_act)
    branch_0 = conv2d_bn(branch_0, block_size * 24, 3,
                         strides=2, padding=padding, activation=base_act)
    branch_1 = conv2d_bn(x, block_size * 16, 1, activation=base_act)
    branch_1 = conv2d_bn(branch_1, block_size * 18, 3,
                         strides=2, padding=padding, activation=base_act)
    branch_2 = conv2d_bn(x, block_size * 16, 1, activation=base_act)
    branch_2 = conv2d_bn(branch_2, block_size * 18, 3, activation=base_act)
    branch_2 = conv2d_bn(branch_2, block_size * 20, 3,
                         strides=2, padding=padding, activation=base_act)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding=padding)(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(
        axis=channel_axis, name=f'{name_prefix}mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): [B 16 16 2080] or 8 x 8 x 2080
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.2,
                                   activation=base_act,
                                   block_type='block8', block_idx=block_idx,
                                   attention_module=attention_module,
                                   name_prefix=name_prefix)
    # Final convolution block: [B 16 16 1536] or 8 x 8 x 1536
    x = conv2d_bn(x, block_size * 96, 1,
                  activation=last_act,
                  name=f'{name_prefix}conv_7b')

    # Create model.
    model = Model(img_input, x,
                  name=f'{name_prefix}inception_resnet_v2')

    return model


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
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
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name)(
            x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation is None:
            pass
        elif activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=ac_name)(x)
        elif activation == 'leakyrelu':
            x = layers.LeakyReLU(0.3, name=ac_name)(x)
        else:
            x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, block_size=16, activation='relu',
                           attention_module="cbam_block", name_prefix=""):
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
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, block_size * 2, 1, activation=activation)
        branch_1 = conv2d_bn(x, block_size * 2, 1, activation=activation)
        branch_1 = conv2d_bn(branch_1, block_size * 2,
                             3, activation=activation)
        branch_2 = conv2d_bn(x, block_size * 2, 1, activation=activation)
        branch_2 = conv2d_bn(branch_2, block_size * 3,
                             3, activation=activation)
        branch_2 = conv2d_bn(branch_2, block_size * 4,
                             3, activation=activation)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, block_size * 12, 1, activation=activation)
        branch_1 = conv2d_bn(x, block_size * 8, 1, activation=activation)
        branch_1 = conv2d_bn(branch_1, block_size * 10,
                             [1, 7], activation=activation)
        branch_1 = conv2d_bn(branch_1, block_size * 12,
                             [7, 1], activation=activation)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, block_size * 12, 1, activation=activation)
        branch_1 = conv2d_bn(x, block_size * 12, 1, activation=activation)
        branch_1 = conv2d_bn(branch_1, block_size * 14,
                             [1, 3], activation=activation)
        branch_1 = conv2d_bn(branch_1, block_size * 16,
                             [3, 1], activation=activation)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = name_prefix + block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(
            branches)
    up = conv2d_bn(
        mixed,
        backend.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_name + '_conv')
    # attention_module
    if attention_module is not None:
        up = attach_attention_module(up, attention_module)

    x = layers.Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=backend.int_shape(x)[1:],
        arguments={'scale': scale},
        name=block_name)([x, up])
    if activation is not None:
        if activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=block_name + '_ac')(x)
        elif activation == 'leakyrelu':
            x = layers.LeakyReLU(0.3, name=block_name + '_ac')(x)
        else:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x
