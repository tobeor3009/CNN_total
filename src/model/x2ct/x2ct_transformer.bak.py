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
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow_addons.layers import InstanceNormalization, AdaptiveAveragePooling2D
from ..inception_resnet_v2_unet_fix.layers import get_act_layer, get_norm_layer
from ..vision_transformer import utils, transformer_layers, swin_layers
from ..vision_transformer.util_layers import Conv3DLayer
from ..vision_transformer.transformer_layers import Pixelshuffle3D, to_tuple


def get_submodules():
    return backend, layers, models, keras_utils


NORM_LAYER = "layer"


def convert_1d_2d(x, num_patch):
    H, W = num_patch
    _, L, C = x.get_shape().as_list()
    assert L == np.prod(num_patch)
    x = backend.reshape(x, shape=(-1, H, W, C))
    return x


def avg_pool(x, num_patch, strides):
    H, W = num_patch
    _, L, C = x.get_shape().as_list()
    assert L == np.prod(num_patch)
    x = backend.reshape(x, shape=(-1, H, W, C))
    x = layers.AveragePooling2D((3, 3), strides=strides,
                                padding="same")(x)
    x = backend.reshape(x, shape=(-1, L // (strides ** 2), C))
    return x


def max_pool(x, num_patch, strides):
    H, W = num_patch
    _, L, C = x.get_shape().as_list()
    assert L == np.prod(num_patch)
    x = backend.reshape(x, shape=(-1, H, W, C))
    x = layers.MaxPooling2D((3, 3), strides=strides,
                            padding="same")(x)
    x = backend.reshape(x, shape=(-1, L // (strides ** 2), C))
    return x


def inceptionv3(input_tensor, block_size,
                patch_size, stride_mode, num_mlp, act="gelu",
                swin_v2=False, return_2d=True):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules()
    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_size[0:2],
                                                            patch_size,
                                                            stride_size)
    x = input_tensor
    # Patch extraction
    x = transformer_layers.PatchExtract(patch_size,
                                        stride_size)(x)
    # Embed patches to tokens
    x = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          block_size * 4)(x)

    x = swin_layers.SwinTransformerBlock(block_size * 4, (num_patch_x, num_patch_y), num_heads=2,
                                         window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                         swin_v2=swin_v2, name="depth_1_1")(x)
    x = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 4,
                                        norm=NORM_LAYER, swin_v2=swin_v2,
                                        name="down_1")(x)
    num_patch_x //= 2
    num_patch_y //= 2
    x = swin_layers.SwinTransformerBlock(block_size * 4, (num_patch_x, num_patch_y), num_heads=2,
                                         window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                         swin_v2=swin_v2, fit_dim=True, name="depth_2_1")(x)
    # skip1.shape = [B 256 256 64]
    if return_2d:
        skip1 = convert_1d_2d(x, (num_patch_x, num_patch_y))
    else:
        skip1 = x

    # previous: strides=(2, 2)
    x = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                         window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                         swin_v2=swin_v2, fit_dim=True, name="depth_2_1")(x)
    x = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=2,
                                         window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                         swin_v2=swin_v2, fit_dim=True, name="depth_2_2")(x)
    x = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 16,
                                        norm=NORM_LAYER, swin_v2=swin_v2,
                                        name="down_2")(x)
    num_patch_x //= 2
    num_patch_y //= 2

    # mixed 0: 35 x 35 x 256
    branch1x1 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_1_1")(x)

    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 6, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_2_1")(x)
    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_2_1")(branch5x5)

    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_3_1")(x)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_3_2")(branch3x3dbl)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_3_2")(branch3x3dbl)

    branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
    branch_pool = swin_layers.SwinTransformerBlock(block_size * 4, (num_patch_x, num_patch_y), num_heads=2,
                                                   window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_3_3_1")(branch_pool)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=-1,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_4_1")(x)

    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 6, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_5_1")(x)
    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_5_2")(branch5x5)

    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_6_1")(x)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_6_2")(branch3x3dbl)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_3_6_3")(branch3x3dbl)

    branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
    branch_pool = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                   window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_3_7_1")(branch_pool)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=-1,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_8_1")(x)

    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 6, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_9_1")(x)
    branch5x5 = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=16, shift_size=8, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_3_9_2")(branch5x5)

    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_10_1")(x)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_10_2")(branch3x3dbl)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_10_3")(branch3x3dbl)

    branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
    branch_pool = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                   window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_3_11_1")(branch_pool)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=-1,
        name='mixed2')
# skip2.shape = [B, 128, 128, ?]
    if return_2d:
        skip2 = convert_1d_2d(x, (num_patch_x, num_patch_y))
    else:
        skip2 = x
    # mixed 3: 17 x 17 x 768
    branch3x3 = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 48,
                                                norm=NORM_LAYER, swin_v2=swin_v2,
                                                name="down_3_1")(x)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 8, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_12_1")(x)
    branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 12, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=8, shift_size=4, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_3_12_2")(branch3x3dbl)
    branch3x3dbl = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 12,
                                                   norm=NORM_LAYER, swin_v2=swin_v2,
                                                   name="down_3_2")(branch3x3dbl)

    branch_pool = max_pool(x, (num_patch_x, num_patch_y), strides=2)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=-1,
        name='mixed3')
    num_patch_x //= 2
    num_patch_y //= 2

    # mixed 4: 17 x 17 x 768
    branch1x1 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                 window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_4_14_1")(x)

    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=4,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_4_15_1")(x)
    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=8,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, name="depth_4_15_2")(branch7x7)
    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, name="depth_4_15_3")(branch7x7)

    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=4,
                                                    window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_4_16_1")(x)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=8,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_16_2")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_16_3")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 16, (num_patch_x, num_patch_y), num_heads=8,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_16_4")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_4_16_5")(branch7x7dbl)

    branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
    branch_pool = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                   window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_4_17_1")(branch_pool)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=-1,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for layer_idx, idx in enumerate(range(18, 20)):
        branch1x1 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                     window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_1")(x)

        branch7x7 = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=4,
                                                     window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_2")(x)
        branch7x7 = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=5,
                                                     window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, name=f"depth_4_{idx}_3")(branch7x7)
        branch7x7 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                     window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_4")(branch7x7)

        branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=4,
                                                        window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_5")(x)
        branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=5,
                                                        window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, name=f"depth_4_{idx}_6")(branch7x7dbl)
        branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=2,
                                                        window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, name=f"depth_4_{idx}_7")(branch7x7dbl)
        branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 20, (num_patch_x, num_patch_y), num_heads=5,
                                                        window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, name=f"depth_4_{idx}_8")(branch7x7dbl)
        branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                        window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_9")(branch7x7dbl)

        branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
        branch_pool = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                       window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                       swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_10")(branch_pool)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=-1,
            name='mixed' + str(5 + layer_idx))

    # mixed 7: 17 x 17 x 768
    branch1x1 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                 window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_4_19_1")(x)

    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                 window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_4_20_1")(x)
    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=8,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, name="depth_4_20_2")(branch7x7)
    branch7x7 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                 window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, name="depth_4_20_3")(branch7x7)

    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                    window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, fit_dim=True, name="depth_4_21_1")(x)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=8,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_21_2")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_21_3")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=8,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_21_4")(branch7x7dbl)
    branch7x7dbl = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                    window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                    swin_v2=swin_v2, name="depth_4_21_5")(branch7x7dbl)

    branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
    branch_pool = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                   window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_4_22_1")(branch_pool)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=-1,
        name='mixed7')
    # skip3.shape = [B, 64, 64, ?]
    if return_2d:
        skip3 = convert_1d_2d(x, (num_patch_x, num_patch_y))
    else:
        skip3 = x
    # mixed 8: 8 x 8 x 1280
    branch3x3 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                 window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                 swin_v2=swin_v2, fit_dim=True, name="depth_4_23_1")(x)
    branch3x3 = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 40,
                                                norm=NORM_LAYER, swin_v2=swin_v2,
                                                name="down_4_1")(branch3x3)

    branch7x7x3 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                   window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, fit_dim=True, name="depth_4_24_1")(x)
    branch7x7x3 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=8,
                                                   window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, name="depth_4_24_2")(branch7x7x3)
    branch7x7x3 = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=2,
                                                   window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                   swin_v2=swin_v2, name="depth_4_24_3")(branch7x7x3)
    branch7x7x3 = transformer_layers.PatchMerging((num_patch_x, num_patch_y), block_size * 24,
                                                  norm=NORM_LAYER, swin_v2=swin_v2,
                                                  name="down_4_2")(branch7x7x3)

    branch_pool = max_pool(x, (num_patch_x, num_patch_y), strides=2)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=-1,
        name='mixed8')
    num_patch_x //= 2
    num_patch_y //= 2

    # mixed 9: 8 x 8 x 2048
    for layer_idx, idx in enumerate(range(25, 28)):
        branch1x1 = swin_layers.SwinTransformerBlock(block_size * 40, (num_patch_x, num_patch_y), num_heads=8,
                                                     window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_1")(x)

        branch3x3 = swin_layers.SwinTransformerBlock(block_size * 48, (num_patch_x, num_patch_y), num_heads=8,
                                                     window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                     swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_2")(x)
        branch3x3_1 = swin_layers.SwinTransformerBlock(block_size * 48, (num_patch_x, num_patch_y), num_heads=4,
                                                       window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                       swin_v2=swin_v2, name=f"depth_4_{idx}_3")(branch3x3)
        branch3x3_2 = swin_layers.SwinTransformerBlock(block_size * 48, (num_patch_x, num_patch_y), num_heads=16,
                                                       window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                       swin_v2=swin_v2, name=f"depth_4_{idx}_4")(branch3x3)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=-1,
            name='mixed9_' + str(idx))

        branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 56, (num_patch_x, num_patch_y), num_heads=8,
                                                        window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_5")(x)
        branch3x3dbl = swin_layers.SwinTransformerBlock(block_size * 56, (num_patch_x, num_patch_y), num_heads=8,
                                                        window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                        swin_v2=swin_v2, name=f"depth_4_{idx}_6")(branch3x3dbl)
        branch3x3dbl_1 = swin_layers.SwinTransformerBlock(block_size * 56, (num_patch_x, num_patch_y), num_heads=4,
                                                          window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                          swin_v2=swin_v2, name=f"depth_4_{idx}_7")(branch3x3dbl)
        branch3x3dbl_2 = swin_layers.SwinTransformerBlock(block_size * 56, (num_patch_x, num_patch_y), num_heads=16,
                                                          window_size=4, shift_size=2, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                          swin_v2=swin_v2, name=f"depth_4_{idx}_8")(branch3x3dbl)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1,
                                           branch3x3dbl_2], axis=-1)

        branch_pool = avg_pool(x, (num_patch_x, num_patch_y), strides=1)
        branch_pool = swin_layers.SwinTransformerBlock(block_size * 24, (num_patch_x, num_patch_y), num_heads=4,
                                                       window_size=2, shift_size=1, num_mlp=num_mlp, act=act, norm=NORM_LAYER,
                                                       swin_v2=swin_v2, fit_dim=True, name=f"depth_4_{idx}_9")(x)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=-1,
            name='mixed' + str(9 + layer_idx))
    # x.shape = [B, 32, 32, 32]
    # skip1.shape = [B 256, 256, 64]
    # skip2.shape = [B 128, 128, ?]
    # skip3.shape = [B 64, 64, ?]
    skip_list = [skip3, skip2, skip1]
    if return_2d:
        x = convert_1d_2d(x, (num_patch_x, num_patch_y))

    return x, skip_list


class PatchExpanding3D(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, norm="layer",
                 swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = to_tuple(upsample_rate)
        self.return_vector = return_vector
        self.norm = get_norm_layer(norm)
        self.swin_v2 = swin_v2
        self.pixel_shuffle_3d = Pixelshuffle3D(kernel_size=upsample_rate)
        self.linear_trans = Conv3DLayer(embed_dim // 2,
                                        kernel_size=1, use_bias=False, use_sn=use_sn,
                                        name='{}_pixel_shuffle_linear_trans'.format(name))

        self.prefix = name

    def call(self, x):
        # rearange depth to number of patches
        x = self.pixel_shuffle_3d(x)
        if self.swin_v2:
            x = self.linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.linear_trans(x)

        return x


class SkipUpsample3DProgressive(layers.Layer):
    def __init__(self, filters, out_filters, z,
                 norm="instance", activation="gelu", naive=True):
        super().__init__()
        self.act_layer = get_act_layer(activation)
        self.norm_layer = get_norm_layer(norm)
        self.upsample_z_num = int(math.log(z, 2))
        for idx in range(self.upsample_z_num):

            if idx == 0:
                conv_kernel_size = (1, 3, 3)
            elif idx < self.upsample_z_num - 3:
                conv_kernel_size = (2, 3, 3)
            elif idx < self.upsample_z_num - 1:
                conv_kernel_size = (3, 3, 3)
            elif idx == self.upsample_z_num - 1:
                conv_kernel_size = 3

            if naive:
                if idx in [0, self.upsample_z_num - 1]:
                    upsample_block = Pixelshuffle3D(kernel_size=(2, 1, 1))
                else:
                    upsample_block = Pixelshuffle3D(kernel_size=(2, 1, 1))
            else:
                upsample_block = Pixelshuffle3D(kernel_size=(2, 1, 1))

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
    decode_shape = [input_shape[0] // 4 for _ in range(3)]

    encoded, skip_list = inceptionv3(model_input, block_size=8,
                                     patch_size=(4, 4), stride_mode="same", num_mlp=512)
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

    decoded = PatchExpanding3D(num_patch=decode_shape,
                               embed_dim=decode_filter_list[-1],
                               upsample_rate=4,
                               swin_v2=True,
                               return_vector=False,
                               name="last_expand")(decoded)
    decoded = layers.Conv3D(last_channel_num, 3, 1, padding="same",
                            use_bias=False)(decoded)
    decoded = get_act_layer(last_act)(decoded)

    return Model(model_input, decoded)


def disc_model(input_shape, last_act):
    model_input = layers.Input(input_shape)

    encoded, skip_list = inceptionv3(model_input, block_size=8,
                                     patch_size=(4, 4), stride_mode="same", num_mlp=512)

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
    encoded, skip_list = inceptionv3(model_input, block_size=8,
                                     patch_size=(4, 4), stride_mode="same", num_mlp=512)

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
