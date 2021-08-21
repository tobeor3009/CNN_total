from keras_applications import get_submodules_from_kwargs

import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
if tf.__version__ >= '2.5.0':
    from . inceptionv3_tf_2_5 import InceptionV3
else:
    from . inceptionv3_tf_2_2 import InceptionV3
# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------


# def get_submodules():
#     return {
#         'backend': backend,
#         'models': models,
#         'layers': layers,
#         'utils': keras_utils,
#     }

def get_submodules():
    return backend, layers, models, keras_utils


def freeze_model(model):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules()
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return


def get_feature_layers(backbone_name, n=5):
    _default_feature_layers = {

        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.

        # VGG
        'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
        'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),

        # ResNets
        'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # ResNeXt
        'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # Inception
        'inceptionv3': (228, 86, 16, 9),
        'inceptionresnetv2': (594, 260, 16, 9),

        # DenseNet
        'densenet121': (311, 139, 51, 4),
        'densenet169': (367, 139, 51, 4),
        'densenet201': (479, 139, 51, 4),

        # SE models
        'seresnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'seresnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'seresnet50': (246, 136, 62, 4),
        'seresnet101': (552, 136, 62, 4),
        'seresnet152': (858, 208, 62, 4),
        'seresnext50': (1078, 584, 254, 4),
        'seresnext101': (2472, 584, 254, 4),
        'senet154': (6884, 1625, 454, 12),

        # Mobile Nets
        'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
        'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                        'block_1_expand_relu'),

        # EfficientNets
        'efficientnetb0': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb3': ('block6a_expand_activation', 'block4a_exp    and_activation',
                           'block3a_expand_activation', 'block2a_exp    and_activation'),
        'efficientnetb4': ('block6a_expand_activation', 'block4a_exp    and_activation',
                           'block3a_expand_activation', 'block2a_exp    and_activation'),
        'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),

    }

    return _default_feature_layers[backbone_name][:n]
# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------


def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        name=None
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    backend, layers, models, keras_utils = get_submodules()
    block_name = name
    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(
            activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(
                axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(
                axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(
            decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='inceptionv3',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        normalization="BatchNormalization",
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    global backend, layers, models, keras_utils

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = InceptionV3(
        include_top=False,
        weights=encoder_weights,
        input_tensor=None,
        input_shape=input_shape,
        classes=None,
        normliazation=normalization,
        pooling=None,
        classifier_activation=None
    )

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
