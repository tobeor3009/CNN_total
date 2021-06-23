# this model avgpool(pool_size=2, stride=2)-conv(kernel_size=1, stride=1)
# when down_sampled residual block

import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, HeNormal, GlorotNormal
from tensorflow.python.keras.layers import normalization


default_initializer = RandomNormal(mean=0.0, stddev=0.02)
RELU_INITIALIZER = HeNormal()
# Xavier
NON_LINEAR_INITIALIZER = GlorotNormal()
NEGATIVE_RATIO = 0.25
# in batchnomalization, gamma_initializer default is "ones", but if use with residual shortcut,
# set to 0 this value will helpful for Early training
GAMMA_INITIALIZER = "zeros"
HIGHWAY_INIT_BIAS = -3
SUBPIXEL_LAYER_SCLAE = 2

# PixelShuffleLayer = Lambda(
#     lambda x: tf.nn.depth_to_space(x, SUBPIXEL_LAYER_SCLAE))


def conv2d_bn(
    x,
    filters,
    kernel_size,
    weight_decay=1e-2,
    strides=(1, 1),
    use_pooling_layer=False,
    activation="leakyrelu",
    normalization=True
):
    if use_pooling_layer:
        cnn_strides = (1, 1)
        pooling_stride = strides
    else:
        cnn_strides = strides

    if activation == "leakyrelu":
        kernel_initializer = RELU_INITIALIZER
    elif activation == "tanh":
        kernel_initializer = NON_LINEAR_INITIALIZER
    else:
        kernel_initializer = NON_LINEAR_INITIALIZER
    param_dict = {
        "kernel_size": kernel_size,
        "padding": "same",
        "use_bias": False,
        "kernel_regularizer": l2(weight_decay),
        "kernel_initializer": kernel_initializer
    }

    layer = Conv2D(filters=filters, strides=cnn_strides, **param_dict)(x)
    if normalization:
        layer = BatchNormalization(axis=-1)(layer)

    if activation == "leakyrelu":
        layer = LeakyReLU(NEGATIVE_RATIO)(layer)
    elif activation == "tanh":
        layer = tanh(layer)

    if use_pooling_layer:
        layer = AveragePooling2D(
            pool_size=pooling_stride,
            strides=pooling_stride,
            padding="same"
        )(layer)
    return layer


def residual_block(
    x,
    filters,
    kernel_size,
    weight_decay=0.0,
    downsample=True,
    use_pooling_layer=False,
    activation="leakyrelu",
    normalization=True,
    Last=False
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(
            pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            residual, filters, kernel_size=1, strides=1, activation=None
        )

    else:
        stride = 1
        if Last:
            residual = conv2d_bn(
                x=x,
                filters=filters,
                kernel_size=3,
                strides=1,
                activation=None,
                normalization=normalization,
            )
        else:
            residual = x
    transform_gate_output = transform_gate(residual, filters)

    conved = conv2d_bn(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        weight_decay=weight_decay,
        strides=stride,
        use_pooling_layer=use_pooling_layer,
        activation=None
    )
    conved = conv2d_bn(
        x=conved,
        filters=filters,
        kernel_size=kernel_size,
        weight_decay=weight_decay,
        strides=1,
        activation=None,
        normalization=normalization,
    )

    output = conved * transform_gate_output + \
        residual * (1 - transform_gate_output)
    if activation == "leakyrelu":
        output = LeakyReLU(NEGATIVE_RATIO)(output)
    elif activation == "tanh":
        output = tanh(output)
    return output


def residual_block_last(
    x,
    filters,
    kernel_size,
    weight_decay=0.0,
    downsample=True,
    activation="sigmoid",
    normalization=True
):
    output = residual_block(
        x, filters, kernel_size, weight_decay, downsample, Last=True,
        activation=None, normalization=normalization
    )
    # use if you want output range [0,1]
    if activation == "sigmoid":
        output = LeakyReLU(NEGATIVE_RATIO)(output)
        output = sigmoid(output)
    # use if you want output range [-1,1]
    elif activation == "tanh":
        output = tanh(output)
    return output


# def pixel_shuffle_block(
#     x,
#     filters,
#     kernel_size,
#     weight_decay=0.0,
#     use_pooling_layer=False,
#     normalization=True,
# ):
#     stride = 1

#     conved_1 = conv2d_bn(
#         x=x,
#         filters=filters,
#         kernel_size=kernel_size,
#         weight_decay=weight_decay,
#         strides=stride,
#         use_pooling_layer=use_pooling_layer,
#         activation=None,
#         normalization=normalization,
#     )
#     conved_2 = conv2d_bn(
#         x=conved_1,
#         filters=filters,
#         kernel_size=kernel_size,
#         weight_decay=weight_decay,
#         strides=stride,
#         use_pooling_layer=use_pooling_layer,
#         activation=None,
#         normalization=normalization,
#     )
#     pixel_shuffled = PixelShuffleLayer(inputs=conved_2)

#     return pixel_shuffled


def deconv2d(
    layer_input,
    skip_input,
    filters,
    kernel_size=3,
    weight_decay=1e-2,
    upsample=True,
    use_upsampling_layer=False,
):
    """Layers used during upsampling"""

    strides = 2 if upsample else 1
    if use_upsampling_layer:
        layer_input = UpSampling2D(
            size=strides, interpolation="bilinear")(layer_input)
    else:
        layer_input = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            # Batch Normalization Beta replace Conv2D Bias
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer=RELU_INITIALIZER
        )(layer_input)
        layer_input = BatchNormalization(axis=-1)(layer_input)
        layer_input = LeakyReLU(NEGATIVE_RATIO)(layer_input)
    concatenated = Concatenate()([layer_input, skip_input])
    output = conv2d_bn(
        x=concatenated,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
    )
    return output


def deconv2d_simple(
    layer_input,
    filters,
    kernel_size=3,
    weight_decay=1e-2,
    upsample=True,
    use_upsampling_layer=False,
):
    """Layers used during upsampling"""

    strides = 2 if upsample else 1
    if use_upsampling_layer:
        layer_input = UpSampling2D(
            size=strides, interpolation="bilinear")(layer_input)
    else:
        layer_input = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            # Batch Normalization Beta replace Conv2D Bias
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer=RELU_INITIALIZER
        )(layer_input)
    output = conv2d_bn(
        x=layer_input,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
    )
    return output


def transform_gate(x, highway_dim):

    transform = Dense(
        units=highway_dim,
        kernel_initializer=NON_LINEAR_INITIALIZER,
        bias_initializer=tf.constant_initializer(HIGHWAY_INIT_BIAS)
    )(x)
    transform = sigmoid(transform)
    return transform
