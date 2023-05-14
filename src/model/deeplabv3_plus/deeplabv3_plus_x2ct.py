# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU
MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras
# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D, ZeroPadding3D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling3D
from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.keras import backend as keras_backend
from ..inception_resnet_v2_unet_fix.layers import get_act_layer

DROPOUT_RATIO = 0.5
BASE_ACT = "gelu"


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides, padding=padding,
                      use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_axis = 1 if keras_backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = InstanceNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = get_act_layer(activation, ac_name)(x)
    return x


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = layers.Conv3D(filters, kernel_size,
                      strides=strides, padding=padding,
                      use_bias=use_bias, name=name)(x)
    if not use_bias:
        bn_axis = 1 if keras_backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = InstanceNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = get_act_layer(activation, ac_name)(x)
    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = InstanceNormalization(name=prefix + '_depthwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = InstanceNormalization(name=prefix + '_pointwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)

    return x


def SepConv_BN_3D(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding3D([(pad_beg, pad_end) for _ in range(3)])(x)
        depth_padding = 'valid'

    _, Z, H, W, C = tf.keras.backend.int_shape(x)
    if C % filters != 0:
        groups = math.gcd(C, filters)
    else:
        groups = filters
    if not depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv3D(filters, (kernel_size, kernel_size, kernel_size),
               strides=(stride, stride, stride),
               dilation_rate=(rate, rate, rate),
               padding=depth_padding, use_bias=False, groups=groups,
               name=prefix + '_depthwise')(x)
    x = InstanceNormalization(name=prefix + '_depthwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv3D(filters, (1, 1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = InstanceNormalization(name=prefix + '_pointwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)

    return x


def SepConv_BN_2D_3D(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding3D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    _, H, W, C = tf.keras.backend.int_shape(x)
    x = layers.Reshape((1, H, W, C))(x)
    x = tf.tile(x, (1, H, 1, 1, 1))
    if not depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv3D(filters, (kernel_size, kernel_size, kernel_size),
               strides=(stride, stride, stride), dilation_rate=(
                   rate, rate, rate),
               padding=depth_padding, use_bias=False, groups=filters,
               name=prefix + '_depthwise')(x)
    x = InstanceNormalization(name=prefix + '_depthwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv3D(filters, (1, 1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = InstanceNormalization(name=prefix + '_pointwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)

    return x


def SepConv_BN_3D_2D(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding3D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    _, Z, H, W, C = tf.keras.backend.int_shape(x)
    x = layers.Permute((2, 3, 4, 1))(x)
    x = layers.Reshape((H, W, C * Z))(x)
    x = conv2d_bn(x, filters, (3, 3), (1, 1),
                  activation=BASE_ACT, use_bias=False)
    if not depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv2D(filters, (kernel_size, kernel_size),
               strides=(stride, stride), dilation_rate=(rate, rate),
               padding=depth_padding, use_bias=False, groups=filters,
               name=prefix + '_depthwise')(x)
    x = InstanceNormalization(name=prefix + '_depthwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = InstanceNormalization(name=prefix + '_pointwise_BN',
                              epsilon=epsilon)(x)
    if depth_activation:
        x = get_act_layer(BASE_ACT)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _conv3d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv3D(filters,
                      (kernel_size, kernel_size, kernel_size),
                      strides=(stride, stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding3D([(pad_beg, pad_end) for _ in range(3)])(x)
        return Conv3D(filters,
                      (kernel_size, kernel_size, kernel_size),
                      strides=(stride, stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = InstanceNormalization(
            name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _xception_block_3d(inputs, depth_list, prefix, skip_connection_type, stride,
                       rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN_3D(residual,
                                 depth_list[i],
                                 prefix + '_separable_conv{}'.format(i + 1),
                                 stride=stride if i == 2 else 1,
                                 rate=rate,
                                 depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv3d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = InstanceNormalization(
            name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Deeplabv3_X2CT(input_tensor=None, input_shape=(256, 256, 4),
                   block_size=16, base_act="gelu", last_act=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (1, 2, 4)
    # img_input.shape = [B, 512, 512, 3]
    x = conv2d_bn(img_input, block_size * 2, (3, 3), (2, 2),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_1')
    # x.shape = [B, 256, 256, 32]
    x = _conv2d_same(x, block_size * 4, 'entry_flow_conv1_2',
                     kernel_size=3, stride=1)
    x = InstanceNormalization(name='entry_flow_conv1_2_BN')(x)
    x = get_act_layer(base_act)(x)
    # x.shape = [B, 256, 256, 64]
    x = _xception_block(x, [block_size * 8 for _ in range(3)], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    # x.shape = [B, 128, 128, 128]
    x, skip1 = _xception_block(x, [block_size * 16 for _ in range(3)], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)
    # x.shape = [B, 64, 64, 256]
    # skip1.shape = [B, 128, 128, 256]
    x = _xception_block(x, [block_size * 45 for _ in range(3)], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    # x.shape = [B, 32, 32, 728]
    for i in range(16):
        x = _xception_block(x, [block_size * 45 for _ in range(3)], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)
    # x.shape = [B, 32, 32, 728]
    x = _xception_block(x, [block_size * 45, block_size * 64, block_size * 64],
                        'exit_flow_block1', skip_connection_type='conv',
                        stride=1, rate=exit_block_rates[0], depth_activation=False)
    # x.shape = [B, 32, 32, 1024]
    x = _xception_block(x, [block_size * 96, block_size * 96, block_size * 128],
                        'exit_flow_block2', skip_connection_type='none',
                        stride=1, rate=exit_block_rates[1], depth_activation=True)
    # x.shape = [B, 32, 32, 2048]
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = GlobalAveragePooling2D()(x)
    # b4.shape = [B, 2048]
    x = SepConv_BN_2D_3D(x, block_size * 64, 'block_2d_3d',
                         rate=1, depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 32, 32, 32, 1024]
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, 1, channels)
    b4 = Reshape((1, 1, 1, b4_shape[1]))(b4)
    # b4.shape = [B, 1, 1, 1, 2048]
    b4 = conv3d_bn(b4, block_size * 16, (1, 1, 1), (1, 1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='image_pooling')
    # b4.shape = [B, 1, 1, 1, 256]
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.tile(b4, (1, *size_before[1:4], 1))
    # b4.shape = [B, 32, 32, 32, 256]
    # simple 1x1
    b0 = conv3d_bn(x, block_size * 16, (1, 1, 1), (1, 1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='aspp0')
    # b0.shape = [B, 32, 32, 32, 256]
    # rate = 6 (12)
    b1 = SepConv_BN_3D(x, block_size * 16, 'aspp1',
                       rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # b1.shape = [B, 32, 32, 32, 256]
    # rate = 12 (24)
    b2 = SepConv_BN_3D(x, block_size * 16, 'aspp2',
                       rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # b2.shape = [B, 32, 32, 32, 256]
    # rate = 18 (36)
    b3 = SepConv_BN_3D(x, block_size * 16, 'aspp3',
                       rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # b3.shape = [B, 32, 32, 32, 256]
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # x.shape = [B, 32, 32, 32, 1280]
    x = conv3d_bn(x, block_size * 16, (1, 1, 1), (1, 1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='concat_projection')
    # x.shape = [B, 32, 32, 32, 256]
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection 1
    # x4 (x2) block
    # skip1.shape = [B, 128, 128, 256]
    x = UpSampling3D(size=(4, 4, 4))(x)
    # x.shape = [B, 128, 128, 128, 256]
    dec_skip1 = SepConv_BN_2D_3D(skip1, block_size * 16, 'block_skip_2d_3d',
                                 rate=1, depth_activation=True, epsilon=1e-5)
    # dec_skip1.shape = [B, 128, 128, 128, 256]
    dec_skip1 = conv3d_bn(dec_skip1, block_size * 3, (1, 1, 1), (1, 1, 1),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection0')
    # dec_skip1.shape = [B, 128, 128, 128, 48]
    x = Concatenate()([x, dec_skip1])
    # x.shape = [B, 128, 128, 128, 304]
    x = SepConv_BN_3D(x, block_size * 16, 'decoder_conv0',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 128, 256]
    x = SepConv_BN_3D(x, block_size * 16, 'decoder_conv1',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 128, 256]

    # Feature projection 2
    # x4 (x2) block
    x = UpSampling3D(size=(2, 2, 2))(x)
    # x.shape = [B, 256, 256, 256, 256]
    x = SepConv_BN_3D(x, block_size * 8, 'decoder_conv2',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 256, 256, 256, 128]
    x = SepConv_BN_3D(x, block_size * 8, 'decoder_conv3',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 256, 256, 256, 128]

    # Feature projection 3
    # x4 (x2) block
    x = UpSampling3D(size=(2, 2, 2))(x)
    # x.shape = [B, 512, 512, 512, 128]
    x = SepConv_BN_3D(x, block_size * 4, 'decoder_conv4',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 512, 512, 512, 64]
    x = SepConv_BN_3D(x, block_size * 4, 'decoder_conv5',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 512, 512, 512, 64]

    last_layer_name = 'logits_semantic'
    x = Conv3D(1, (1, 1, 1), padding='same', name=last_layer_name)(x)
    # x.shape = [B, 512, 512, 512, 1]
    # img_input.shape = [B, 512, 512, 3]
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = get_act_layer(last_act)(x)
    model = Model(inputs, x, name='deeplabv3plus_x2ct')
    return model


def Deeplabv3_CT2X(input_tensor=None, input_shape=(256, 256, 256, 1), last_channel_num=4,
                   block_size=16, base_act="gelu", last_act=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (1, 2, 4)
    # img_input.shape = [B, 256, 256, 256, 1]
    x = conv3d_bn(img_input, block_size * 2, (3, 3, 3), (2, 2, 2),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_1')
    # x.shape = [B, 128, 128, 128, 32]
    x = _conv3d_same(x, block_size * 4, 'entry_flow_conv1_2',
                     kernel_size=3, stride=1)
    x = InstanceNormalization(name='entry_flow_conv1_2_BN')(x)
    x = get_act_layer(base_act)(x)
    # x.shape = [B, 128, 128, 128, 64]
    x = _xception_block_3d(x, [block_size * 4 for _ in range(3)], 'entry_flow_block1',
                           skip_connection_type='conv', stride=2,
                           depth_activation=False)
    # x.shape = [B, 64, 64, 64, 64]
    x, skip1 = _xception_block_3d(x, [block_size * 8 for _ in range(3)], 'entry_flow_block2',
                                  skip_connection_type='conv', stride=2,
                                  depth_activation=False, return_skip=True)
    # x.shape = [B, 32, 32, 32, 128]
    # skip1.shape = [B, 64, 64, 64, 128]
    x = _xception_block_3d(x, [block_size * 16 for _ in range(3)], 'entry_flow_block3',
                           skip_connection_type='conv', stride=entry_block3_stride,
                           depth_activation=False)
    # x.shape = [B, 16, 16, 16, 256]
    for i in range(8):
        x = _xception_block_3d(x, [block_size * 16 for _ in range(3)], 'middle_flow_unit_{}'.format(i + 1),
                               skip_connection_type='sum', stride=1, rate=middle_block_rate,
                               depth_activation=False)
    # x.shape = [B, 16, 16, 16, 256]
    x = _xception_block_3d(x, [block_size * 16, block_size * 32, block_size * 32],
                           'exit_flow_block1', skip_connection_type='conv',
                           stride=1, rate=exit_block_rates[0], depth_activation=False)
    # x.shape = [B, 16, 16, 16, 1024]
    x = _xception_block_3d(x, [block_size * 48, block_size * 48, block_size * 64],
                           'exit_flow_block2', skip_connection_type='none',
                           stride=1, rate=exit_block_rates[1], depth_activation=True)
    # x.shape = [B, 16, 16, 16, 1024]
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = GlobalAveragePooling3D()(x)
    x = SepConv_BN_3D_2D(x, block_size * 64, 'block_3d_2d',
                         rate=1, depth_activation=True, epsilon=1e-5)
    # b4.shape = [B, 2048]
    # x.shape = [B, 16, 16, 1024]
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    # b4.shape = [B, 1, 1, 2048]
    b4 = conv2d_bn(b4, block_size * 16, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='image_pooling')
    # b4.shape = [B, 1, 1, 256]
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.tile(b4, (1, *size_before[1:3], 1))
    # b4.shape = [B, 16, 16, 256]
    # simple 1x1
    b0 = conv2d_bn(x, block_size * 16, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='aspp0')
    # b0.shape = [B, 16, 16, 256]
    # rate = 6 (12)
    b1 = SepConv_BN(x, block_size * 16, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # b1.shape = [B, 16, 16, 256]
    # rate = 12 (24)
    b2 = SepConv_BN(x, block_size * 16, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # b2.shape = [B, 16, 16, 256]
    # rate = 18 (36)
    b3 = SepConv_BN(x, block_size * 16, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # b3.shape = [B, 16, 16, 256]
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # x.shape = [B, 16, 16, 1280]
    x = conv2d_bn(x, block_size * 16, (1, 1), (1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='concat_projection')
    # x.shape = [B, 16, 16, 256]
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection 1
    # x4 (x2) block
    # skip1.shape = [B, 64, 64, 256]
    x = UpSampling2D(size=(4, 4))(x)
    # x.shape = [B, 64, 64, 256]
    dec_skip1 = SepConv_BN_3D_2D(skip1, block_size * 16, 'block_skip_2d_3d',
                                 rate=1, depth_activation=True, epsilon=1e-5)
    # dec_skip1.shape = [B, 64, 64, 256]
    dec_skip1 = conv2d_bn(dec_skip1, block_size * 3, (1, 1), (1, 1),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection0')
    # dec_skip1.shape = [B, 64, 64, 48]
    x = Concatenate()([x, dec_skip1])
    # x.shape = [B, 64, 64, 304]
    x = SepConv_BN(x, block_size * 16, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 256]
    x = SepConv_BN(x, block_size * 16, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 256]

    # Feature projection 2
    # x4 (x2) block
    x = UpSampling2D(size=(2, 2))(x)
    # x.shape = [B, 64, 64, 256]
    x = SepConv_BN(x, block_size * 8, 'decoder_conv2',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 128]
    x = SepConv_BN(x, block_size * 8, 'decoder_conv3',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 128]

    # Feature projection 3
    # x4 (x2) block
    x = UpSampling3D(size=(2, 2, 2))(x)
    # x.shape = [B, 64, 64, 128]
    x = SepConv_BN(x, block_size * 4, 'decoder_conv4',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 64]
    x = SepConv_BN(x, block_size * 4, 'decoder_conv5',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 64, 64, 64]

    last_layer_name = 'logits_semantic'
    x = Conv2D(last_channel_num, (1, 1),
               padding='same', name=last_layer_name)(x)
    # x.shape = [B, 512, 512, 512, 1]
    # img_input.shape = [B, 512, 512, 3]
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = get_act_layer(last_act)(x)
    model = Model(inputs, x, name='deeplabv3plus_ct2x')
    return model


def Deeplabv3_DISC_3D(input_tensor=None, input_shape=(256, 256, 256, 1),
                      block_size=16, base_act="gelu", last_act=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (1, 2, 4)
    # img_input.shape = [B, 256, 256, 256, 1]
    x = conv3d_bn(img_input, block_size * 2, (3, 3, 3), (2, 2, 2),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_1')
    # x.shape = [B, 128, 128, 128, 32]
    x = _conv3d_same(x, block_size * 4, 'entry_flow_conv1_2',
                     kernel_size=3, stride=1)
    x = InstanceNormalization(name='entry_flow_conv1_2_BN')(x)
    x = get_act_layer(base_act)(x)
    # x.shape = [B, 128, 128, 128, 64]
    x = _xception_block_3d(x, [block_size * 4 for _ in range(3)], 'entry_flow_block1',
                           skip_connection_type='conv', stride=2,
                           depth_activation=False)
    # x.shape = [B, 64, 64, 64, 64]
    x, skip1 = _xception_block_3d(x, [block_size * 8 for _ in range(3)], 'entry_flow_block2',
                                  skip_connection_type='conv', stride=2,
                                  depth_activation=False, return_skip=True)
    # x.shape = [B, 32, 32, 32, 128]
    # skip1.shape = [B, 64, 64, 64, 128]
    x = _xception_block_3d(x, [block_size * 16 for _ in range(3)], 'entry_flow_block3',
                           skip_connection_type='conv', stride=entry_block3_stride,
                           depth_activation=False)
    # x.shape = [B, 16, 16, 16, 256]
    for i in range(8):
        x = _xception_block_3d(x, [block_size * 16 for _ in range(3)], f'middle_flow_unit_{i + 1}',
                               skip_connection_type='sum', stride=1, rate=middle_block_rate,
                               depth_activation=False)
    # x.shape = [B, 16, 16, 16, 256]
    x = _xception_block_3d(x, [block_size * 16, block_size * 32, block_size * 32],
                           'exit_flow_block1', skip_connection_type='conv',
                           stride=1, rate=exit_block_rates[0], depth_activation=False)
    # x.shape = [B, 16, 16, 16, 1024]
    x = _xception_block_3d(x, [block_size * 48, block_size * 48, block_size * 64],
                           'exit_flow_block2', skip_connection_type='none',
                           stride=1, rate=exit_block_rates[1], depth_activation=True)
    # x.shape = [B, 16, 16, 16, 1024]
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = GlobalAveragePooling3D()(x)
    # b4.shape = [B, 2048]
    # x.shape = [B, 16, 16, 16, 1024]
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, 1, channels)
    b4 = Reshape((1, 1, 1, b4_shape[1]))(b4)
    # b4.shape = [B, 1, 1, 1, 2048]
    b4 = conv3d_bn(b4, block_size * 16, (1, 1, 1), (1, 1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='image_pooling')
    # b4.shape = [B, 1, 1, 1, 256]
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.tile(b4, (1, *size_before[1:4], 1))
    # b4.shape = [B, 16, 16, 16, 256]
    # simple 1x1
    b0 = conv3d_bn(x, block_size * 16, (1, 1, 1), (1, 1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='aspp0')
    # b0.shape = [B, 16, 16, 16, 256]
    # rate = 6 (12)
    b1 = SepConv_BN_3D(x, block_size * 16, 'aspp1',
                       rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # b1.shape = [B, 16, 16, 16, 256]
    # rate = 12 (24)
    b2 = SepConv_BN_3D(x, block_size * 16, 'aspp2',
                       rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # b2.shape = [B, 16, 16, 16, 256]
    # rate = 18 (36)
    b3 = SepConv_BN_3D(x, block_size * 16, 'aspp3',
                       rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # b3.shape = [B, 16, 16, 16, 256]
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # x.shape = [B, 16, 16, 16, 1280]
    x = conv3d_bn(x, block_size * 16, (1, 1, 1), (1, 1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='concat_projection')
    # x.shape = [B, 16, 16, 16, 256]
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection 1
    # x4 (x2) block
    # skip1.shape = [B, 64, 64, 256]
    dec_skip1 = conv3d_bn(skip1, block_size * 8, (3, 3, 3), (2, 2, 2),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection0')
    # dec_skip1.shape = [B, 32, 32, 32, 128]
    dec_skip1 = conv3d_bn(dec_skip1, block_size * 3, (3, 3, 3), (2, 2, 2),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection1')
    # dec_skip1.shape = [B, 16, 16, 16, 48]
    x = Concatenate()([x, dec_skip1])
    # x.shape = [B, 16, 16, 16, 304]
    x = SepConv_BN_3D(x, block_size * 16, 'decoder_conv0',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 16, 16, 16, 256]
    x = SepConv_BN_3D(x, block_size * 16, 'decoder_conv1',
                      depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 16, 16, 16, 256]

    last_layer_name = 'logits_semantic'
    x = Conv3D(1, (1, 1, 1), padding='same', name=last_layer_name)(x)
    # x.shape = [B, 512, 512, 512, 1]
    # img_input.shape = [B, 512, 512, 3]
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = get_act_layer(last_act)(x)
    model = Model(inputs, x, name='deeplabv3plus_disc_3d')
    return model


def Deeplabv3_DISC_2D(input_tensor=None, input_shape=(256, 256, 1),
                      block_size=16, base_act="gelu", last_act=None):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (6, 12, 18)
    # img_input.shape = [B, 512, 512, 3]
    x = conv2d_bn(img_input, block_size * 2, (3, 3), (2, 2),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_1')
    # x.shape = [B, 256, 256, 32]
    x = _conv2d_same(x, block_size * 4, 'entry_flow_conv1_2',
                     kernel_size=3, stride=1)
    x = InstanceNormalization(name='entry_flow_conv1_2_BN')(x)
    x = get_act_layer(base_act)(x)
    # x.shape = [B, 256, 256, 64]
    x = _xception_block(x, [block_size * 8 for _ in range(3)], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    # x.shape = [B, 128, 128, 128]
    x, skip1 = _xception_block(x, [block_size * 16 for _ in range(3)], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)
    # x.shape = [B, 64, 64, 256]
    # skip1.shape = [B, 128, 128, 256]
    x = _xception_block(x, [block_size * 45 for _ in range(3)], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    # x.shape = [B, 32, 32, 728]
    for i in range(16):
        x = _xception_block(x, [block_size * 45 for _ in range(3)], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)
    # x.shape = [B, 32, 32, 728]
    x = _xception_block(x, [block_size * 45, block_size * 64, block_size * 64],
                        'exit_flow_block1', skip_connection_type='conv',
                        stride=1, rate=exit_block_rates[0], depth_activation=False)
    # x.shape = [B, 32, 32, 1024]
    x = _xception_block(x, [block_size * 96, block_size * 96, block_size * 128],
                        'exit_flow_block2', skip_connection_type='none',
                        stride=1, rate=exit_block_rates[1], depth_activation=True)
    # x.shape = [B, 32, 32, 2048]
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = GlobalAveragePooling2D()(x)
    # b4.shape = [B, 2048]
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    # b4.shape = [B, 1, 1, 2048]
    b4 = conv2d_bn(b4, block_size * 16, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='image_pooling')
    # b4.shape = [B, 1, 1, 256]
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.tile(b4, (1, *size_before[1:3], 1))
    # b4.shape = [B, 32, 32, 256]
    # simple 1x1
    b0 = conv2d_bn(x, block_size * 16, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='aspp0')
    # b0.shape = [B, 32, 32, 256]
    # rate = 6 (12)
    b1 = SepConv_BN(x, block_size * 16, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # b1.shape = [B, 32, 32, 256]
    # rate = 12 (24)
    b2 = SepConv_BN(x, block_size * 16, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # b2.shape = [B, 32, 32, 256]
    # rate = 18 (36)
    b3 = SepConv_BN(x, block_size * 16, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # b3.shape = [B, 32, 32, 256]
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # x.shape = [B, 32, 32, 1280]
    x = conv2d_bn(x, block_size * 16, (1, 1), (1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='concat_projection')
    # x.shape = [B, 32, 32, 256]
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection
    # x4 (x2) block
    # skip1.shape = [B, 128, 128, 256]
    skip_size = tf.keras.backend.int_shape(skip1)
    dec_skip1 = conv2d_bn(skip1, block_size * 8, (3, 3), (2, 2),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection0')
    dec_skip1 = conv2d_bn(dec_skip1, block_size * 3, (3, 3), (2, 2),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection1')

    # dec_skip1.shape = [B, 128, 128, 48]
    x = Concatenate()([x, dec_skip1])
    # x.shape = [B, 128, 128, 304]
    x = SepConv_BN(x, block_size * 16, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 256]
    x = SepConv_BN(x, block_size * 16, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 256]
    last_layer_name = 'logits_semantic'

    x = Conv2D(1, (1, 1), padding='same', name=last_layer_name)(x)
    # x.shape = [B, 128, 128, classes]
    # img_input.shape = [B, 512, 512, 3]
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = get_act_layer(last_act)(x)
    model = Model(inputs, x, name='deeplabv3plus_disc_2d')
    return model
