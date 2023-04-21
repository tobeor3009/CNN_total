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

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras import backend as keras_backend
from ..inception_resnet_v2_unet_fix.layers import get_act_layer
from .util import BASE_ACT, RGB_OUTPUT_CHANNEL, SEG_OUTPUT_CHANNEL
DROPOUT_RATIO = 0.5


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


def pixel_shuffle_block(x, skip_list, base_act, last_act):

    # skip1.shape = [B, 512, 512, 32]
    # skip2.shape = [B, 256, 256, 128]
    # skip3.shape = [B, 128, 128, 256]
    # skip4.shape = [B, 64, 64, 728]
    skip1, skip2, skip3, skip4 = skip_list

    def pixel_shuffle_mini_block(x, block_size, skip, idx):

        skip_channel_list = [48, 36, 24, 12]
        x = Conv2D(block_size * 4, (1, 1), padding='same',
                   use_bias=False, name=f'decoder_pointwise_{idx}')(x)
        x = InstanceNormalization(name=f'decoder_pointwise_IN_{idx}',
                                  epsilon=1e-3)(x)
        x = tf.nn.depth_to_space(x, block_size=2)
        dec_skip = conv2d_bn(skip, skip_channel_list[idx - 1], (1, 1), (1, 1),
                             padding="same", activation=base_act,
                             use_bias=False, name=f'feature_projection_{idx}')
        x = Concatenate()([x, dec_skip])
        x = SepConv_BN(x, block_size, f'decoder_conv_{idx}',
                       depth_activation=True, epsilon=1e-5)
        return x
    # x.shape = [B, 32, 32, 256]
    x = pixel_shuffle_mini_block(x, 256, skip4, 1)
    # x.shape = [B, 64, 64, 256]
    x = pixel_shuffle_mini_block(x, 192, skip3, 2)
    # x.shape = [B, 128, 128, 192]
    x = pixel_shuffle_mini_block(x, 128, skip2, 3)
    # x.shape = [B, 256, 256, 128]
    x = pixel_shuffle_mini_block(x, 64, skip1, 4)
    # x.shape = [B, 512, 512, 64]
    # x.shape = [B, 128, 128, 196]
    x = Conv2D(RGB_OUTPUT_CHANNEL, (1, 1),
               padding='same')(x)
    # img_input.shape = [B, 512, 512, 3]
    x = get_act_layer(last_act)(x)
    return x


def upsample_block(x, skip_list, base_act, last_act, class_num):

    # skip1.shape = [B, 512, 512, 32]
    # skip2.shape = [B, 256, 256, 128]
    # skip3.shape = [B, 128, 128, 256]
    # skip4.shape = [B, 64, 64, 728]
    skip1, skip2, skip3, skip4 = skip_list
    x = UpSampling2D(size=(4, 4))(x)
    # x.shape = [B, 128, 128, 256]
    dec_skip3 = conv2d_bn(skip3, 48, (1, 1), (1, 1),
                          padding="same", activation=base_act,
                          use_bias=False, name='feature_projection0')

    # dec_skip1.shape = [B, 128, 128, 48]
    x = Concatenate()([x, dec_skip3])
    # x.shape = [B, 128, 128, 304]
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 256]
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)
    # x.shape = [B, 128, 128, 256]
    last_layer_name = 'logits_semantic'

    x = Conv2D(class_num, (1, 1),
               padding='same', name=last_layer_name)(x)
    # x.shape = [B, 128, 128, classes]
    # img_input.shape = [B, 512, 512, 3]
    x = UpSampling2D(size=(4, 4))(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    x = get_act_layer(last_act)(x)
    return x


def Deeplabv3(input_shape=(512, 512, 3), class_num=1,
              base_act=BASE_ACT, image_last_act="tanh", last_act="sigmoid", multi_task=False):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    img_input = Input(shape=input_shape)
    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rates = (6, 12, 18)
    # img_input.shape = [B, 512, 512, 3]
    x = conv2d_bn(img_input, 32, (3, 3), (1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_1')
    skip1 = x
    x = conv2d_bn(img_input, 32, (3, 3), (2, 2),
                  padding="same", activation=base_act,
                  use_bias=False, name='entry_flow_conv1_2')
    # x.shape = [B, 256, 256, 32]
    x = _conv2d_same(x, 64, 'entry_flow_conv1_3', kernel_size=3, stride=1)
    x = InstanceNormalization(name='entry_flow_conv1_3_IN')(x)
    x = get_act_layer(base_act)(x)
    # x.shape = [B, 256, 256, 64]
    x, skip2 = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)
    # x.shape = [B, 128, 128, 128]
    x, skip3 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)
    # x.shape = [B, 64, 64, 256]
    x, skip4 = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                               skip_connection_type='conv', stride=entry_block3_stride,
                               depth_activation=False, return_skip=True)
    # x.shape = [B, 32, 32, 728]
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)
    # x.shape = [B, 32, 32, 728]
    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    # x.shape = [B, 32, 32, 1024]
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
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
    b4 = conv2d_bn(b4, 256, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='image_pooling')
    # b4.shape = [B, 1, 1, 256]
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.tile(b4, (1, *size_before[1:3], 1))
    # b4.shape = [B, 32, 32, 256]
    # simple 1x1
    b0 = conv2d_bn(x, 256, (1, 1), (1, 1),
                   padding="same", activation=base_act,
                   use_bias=False, name='aspp0')
    # b0.shape = [B, 32, 32, 256]
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # b1.shape = [B, 32, 32, 256]
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # b2.shape = [B, 32, 32, 256]
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # b3.shape = [B, 32, 32, 256]
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    # x.shape = [B, 32, 32, 1280]
    x = conv2d_bn(x, 256, (1, 1), (1, 1),
                  padding="same", activation=base_act,
                  use_bias=False, name='concat_projection')
    # x.shape = [B, 32, 32, 256]
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder
    skip_list = [skip1, skip2, skip3, skip4]
    # skip1.shape = [B, 512, 512, 32]
    # skip2.shape = [B, 256, 256, 128]
    # skip3.shape = [B, 128, 128, 256]
    # skip4.shape = [B, 64, 64, 728]
    seg_output = upsample_block(x, skip_list, base_act, last_act, class_num)
    if multi_task:
        he_output = pixel_shuffle_block(x, skip_list, base_act, image_last_act)
        model = Model(img_input, [he_output, seg_output],
                      name='deeplabv3plus')
    else:
        model = Model(img_input, seg_output, name='deeplabv3plus')
    return model
