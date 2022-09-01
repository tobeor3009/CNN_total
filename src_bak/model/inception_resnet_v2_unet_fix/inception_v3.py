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

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils as keras_utils

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export


def get_submodules():
    return backend, layers, models, keras_utils


WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
SKIP_CONNECTION_LAYER_NAMES = ["pooling1", "pooling2",
                               "mixed1", "mixed3", "mixed8"]


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=1,
              normliazation="BatchNormalization",
              activation="relu",
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    if name is not None:
        if normliazation == "BatchNormalization":
            normalization_name = name + '_bn'
        elif normliazation == "LayerNormalization":
            normalization_name = name + '_ln'
        elif normliazation == "InstanceNormalization":
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
    x = layers.BatchNormalization(
        axis=bn_axis, scale=False, name=normalization_name)(x)
    if activation is None:
        pass
    elif activation == 'relu':
        x = layers.Activation(tf.nn.relu6, name=name)(x)
    elif activation == 'leakyrelu':
        x = layers.LeakyReLU(0.3, name=name)(x)
    else:
        x = layers.Activation(activation, name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                block_size=16,
                pooling=None,
                base_act="relu",
                last_act="relu",
                classes=1000,
                label_input=None,
                label_tensor=None,
                target_label_input=None,
                target_label_tensor=None,
                stargan_mode=None,
                **kwargs):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules()

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    if stargan_mode == "add":
        start_input = img_input + label_tensor
    elif stargan_mode == "concatenate":
        start_input = backend.concatenate(
            [img_input, label_tensor], axis=channel_axis)
    else:
        start_input = img_input

    x = conv2d_bn(start_input, block_size * 2, 3,
                  3, strides=(2, 2), padding='same', activation=base_act, name='pooling1')
    x = conv2d_bn(x, block_size * 2, 3, 3, padding='same', activation=base_act)
    x = conv2d_bn(x, block_size * 4, 3, 3, padding='same', activation=base_act)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pooling2')(x)

    x = conv2d_bn(x, block_size * 5, 1, 1, padding='same', activation=base_act)
    x = conv2d_bn(x, block_size * 12, 3, 3,
                  padding='same', activation=base_act)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)

    branch5x5 = conv2d_bn(x, block_size * 3, 1, 1, activation=base_act)
    branch5x5 = conv2d_bn(branch5x5, block_size * 4, 5, 5, activation=base_act)

    branch3x3dbl = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            2, 1, 1, activation=base_act)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)

    branch5x5 = conv2d_bn(x, block_size * 3, 1, 1, activation=base_act)
    branch5x5 = conv2d_bn(branch5x5, block_size * 4, 5, 5, activation=base_act)

    branch3x3dbl = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            4, 1, 1, activation=base_act)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)

    branch5x5 = conv2d_bn(x, block_size * 3, 1, 1, activation=base_act)
    branch5x5 = conv2d_bn(branch5x5, block_size * 4, 5, 5, activation=base_act)

    branch3x3dbl = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            4, 1, 1, activation=base_act)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, block_size * 24, 3, 3,
                          strides=(2, 2), padding='same', activation=base_act)

    branch3x3dbl = conv2d_bn(x, block_size * 4, 1, 1, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size *
                             6, 3, 3, activation=base_act)
    branch3x3dbl = conv2d_bn(branch3x3dbl, block_size * 6, 3, 3,
                             strides=(2, 2), padding='same', activation=base_act)

    branch_pool = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)

    branch7x7 = conv2d_bn(x, block_size * 8, 1, 1, activation=base_act)
    branch7x7 = conv2d_bn(branch7x7, block_size * 8, 1, 7, activation=base_act)
    branch7x7 = conv2d_bn(branch7x7, block_size * 12,
                          7, 1, activation=base_act)

    branch7x7dbl = conv2d_bn(x, block_size * 8, 1, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             8, 7, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             8, 1, 7, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             8, 7, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             8, 1, 7, activation=base_act)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            12, 1, 1, activation=base_act)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)

        branch7x7 = conv2d_bn(x, block_size * 10, 1, 1, activation=base_act)
        branch7x7 = conv2d_bn(branch7x7, block_size * 10,
                              1, 7, activation=base_act)
        branch7x7 = conv2d_bn(branch7x7, block_size * 12,
                              7, 1, activation=base_act)

        branch7x7dbl = conv2d_bn(x, block_size * 10, 1, 1, activation=base_act)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, block_size * 10, 7, 1, activation=base_act)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, block_size * 10, 1, 7, activation=base_act)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, block_size * 10, 7, 1, activation=base_act)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, block_size * 12, 1, 7, activation=base_act)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(
            branch_pool, block_size * 12, 1, 1, activation=base_act)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)

    branch7x7 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)
    branch7x7 = conv2d_bn(branch7x7, block_size * 12,
                          1, 7, activation=base_act)
    branch7x7 = conv2d_bn(branch7x7, block_size * 12,
                          7, 1, activation=base_act)

    branch7x7dbl = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             12, 7, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             12, 1, 7, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             12, 7, 1, activation=base_act)
    branch7x7dbl = conv2d_bn(branch7x7dbl, block_size *
                             12, 1, 7, activation=base_act)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, block_size *
                            12, 1, 1, activation=base_act)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)
    branch3x3 = conv2d_bn(branch3x3, block_size * 20, 3, 3,
                          strides=(2, 2), padding='same', activation=base_act)

    branch7x7x3 = conv2d_bn(x, block_size * 12, 1, 1, activation=base_act)
    branch7x7x3 = conv2d_bn(branch7x7x3, block_size *
                            12, 1, 7, activation=base_act)
    branch7x7x3 = conv2d_bn(branch7x7x3, block_size *
                            12, 7, 1, activation=base_act)
    branch7x7x3 = conv2d_bn(branch7x7x3, block_size * 12, 3, 3,
                            strides=(2, 2), padding='same', activation=base_act)

    branch_pool = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, block_size * 20, 1, 1, activation=last_act)

        branch3x3 = conv2d_bn(x, block_size * 24, 1, 1, activation=last_act)
        branch3x3_1 = conv2d_bn(
            branch3x3, block_size * 24, 1, 3, activation=last_act)
        branch3x3_2 = conv2d_bn(
            branch3x3, block_size * 24, 3, 1, activation=last_act)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, block_size * 28, 1, 1, activation=last_act)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, block_size * 24, 3, 3, activation=last_act)
        branch3x3dbl_1 = conv2d_bn(
            branch3x3dbl, block_size * 24, 1, 3, activation=last_act)
        branch3x3dbl_2 = conv2d_bn(
            branch3x3dbl, block_size * 24, 3, 1, activation=last_act)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(
            branch_pool, block_size * 12, 1, 1, activation=last_act)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    if stargan_mode == "add":
        x = x + target_label_tensor
    elif stargan_mode == "concatenate":
        x = backend.concatenate([x, target_label_tensor], axis=channel_axis)

    # Create model.
    if stargan_mode is None:
        model = models.Model(inputs, x, name='inception_v3')
    else:
        model = models.Model(
            [inputs, label_input, target_label_input], x, name='inception_v3')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
