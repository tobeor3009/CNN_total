# tensorflow Module
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from .base_model_resnet import HighWayResnet2D
DEFAULT_INITIALIZER = RandomNormal(mean=0.0, stddev=0.02)
KERNEL_SIZE = (3, 3)


def get_resnet_disc(input_shape,
                    base_act="leakyrelu",
                    last_act="sigmoid",
                    num_downsample=5,
                    block_size=16
                    ):
    padding = "same"
    encoder_output_filter = None
    groups = 1
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    base_input = encoder.input

    # add a global spatial average pooling layer
    x = encoder.output
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(1)(x)
    model = Model(base_input, predictions)
    return model


def build_discriminator(input_img_shape, output_img_shape, depth=None,
                        discriminator_power=32,
                        kernel_initializer=DEFAULT_INITIALIZER):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_img = Input(shape=input_img_shape)
    man_or_model_mad_img = Input(shape=output_img_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([original_img, man_or_model_mad_img])

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3
    down_sampled_layer = conv2d_bn(
        x=combined_imgs, filters=discriminator_power,
        kernel_size=(3, 3), kernel_initializer=kernel_initializer,
        weight_decay=1e-4, strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
            weight_decay=1e-4, downsample=False
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
            weight_decay=1e-4, downsample=True, use_pooling_layer=True
        )

    validity = residual_block_last(x=down_sampled_layer, filters=1,
                                   kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
                                   weight_decay=1e-4, downsample=False, activation="sigmoid")

    return Model([original_img, man_or_model_mad_img], validity)


from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


default_initializer = RandomNormal(mean=0.0, stddev=0.02)
NEGATIVE_RATIO = 0.25
SIGMOID_NEGATIVE_RATIO = 0.25
# in batchnomalization, gamma_initializer default is "ones", but if use with residual shortcut,
# set to 0 this value will helpful for Early training
GAMMA_INITIALIZER = "zeros"


def conv2d_bn(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    strides=(1, 1),
    use_pooling_layer=False
):
    if use_pooling_layer:
        cnn_strides = (1, 1)
        pooling_stride = strides
    else:
        cnn_strides = strides

    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=cnn_strides,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer=kernel_initializer,
    )(x)
    layer = BatchNormalization(axis=-1)(layer)
    layer = LeakyReLU(NEGATIVE_RATIO)(layer)
    if use_pooling_layer:
        layer = MaxPooling2D(pool_size=pooling_stride,
                             strides=pooling_stride, padding="same")(layer)
    return layer


def residual_block(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    use_pooling_layer=False
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(
            pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            residual, filters,
            kernel_size=1, kernel_initializer=kernel_initializer,
            strides=1
        )
    else:
        residual = x
        stride = 1

    conved = conv2d_bn(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=stride,
        use_pooling_layer=use_pooling_layer
    )
    conved = conv2d_bn(
        x=conved,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=1
    )
    output = layers.add([conved, residual])
    output = BatchNormalization(
        axis=-1, gamma_initializer=GAMMA_INITIALIZER)(output)
    output = LeakyReLU(NEGATIVE_RATIO)(output)

    return output


def wide_residual_block(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    width=4,
    downsample=True,
    use_pooling_layer=False,

):
    if downsample:
        stride = 2
        residual = AveragePooling2D(
            pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            x=residual, filters=filters,
            kernel_size=1, kernel_initializer=kernel_initializer,
            strides=1
        )
    else:
        residual = x
        stride = 1
    conved_stacked = []
    for _ in range(width):
        conved = conv2d_bn(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            weight_decay=weight_decay,
            strides=stride,
            use_pooling_layer=use_pooling_layer
        )
        conved = conv2d_bn(
            x=conved,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            weight_decay=weight_decay,
            strides=1
        )
        added_layer = layers.add([conved, residual])
        added_layer = BatchNormalization(
            axis=-1, gamma_initializer=GAMMA_INITIALIZER)(added_layer)
        added_layer = LeakyReLU(NEGATIVE_RATIO)(added_layer)
        conved_stacked.append(added_layer)

    conved_stacked = layers.concatenate(conved_stacked)
    conved_stacked = BatchNormalization(axis=-1)(conved_stacked)
    conved_stacked = LeakyReLU(NEGATIVE_RATIO)(conved_stacked)

    return conved_stacked


def residual_block_last(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    activation='sigmoid'
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(pool_size=stride)(x)
        residual = conv2d_bn(
            x=residual, filters=filters,
            kernel_size=1, kernel_initializer=kernel_initializer,
            strides=1
        )
    else:
        residual = conv2d_bn(
            x=x, filters=filters,
            kernel_size=1, kernel_initializer=kernel_initializer,
            strides=1
        )
        stride = 1

    conved = conv2d_bn(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=stride,
    )
    conved = conv2d_bn(
        x=conved,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=1,
    )
    output = layers.add([conved, residual])
    output = BatchNormalization(
        axis=-1, gamma_initializer=GAMMA_INITIALIZER)(output)
    output = LeakyReLU(SIGMOID_NEGATIVE_RATIO)(output)
    # use if you want output range [0,1]
    if activation == 'sigmoid':
        output = sigmoid(output)
    # use if you want output range [-1,1]
    elif activation == 'tanh':
        output = tanh(output)
    return output


def deconv2d(
    layer_input,
    skip_input,
    filters,
    kernel_size=3,
    kernel_initializer=default_initializer,
    upsample=True,
    use_upsampling_layer=False


):
    """Layers used during upsampling"""

    strides = 2 if upsample else 1

    if use_upsampling_layer:
        layer_input = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            kernel_regularizer=l2(0.0),
            kernel_initializer=kernel_initializer
        )(layer_input)
        layer_input = UpSampling2D(
            size=strides, interpolation='nearest')(layer_input)
    else:
        layer_input = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_regularizer=l2(0.0),
            kernel_initializer=kernel_initializer
        )(layer_input)
    layer_input = conv2d_bn(
        x=layer_input,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        strides=1,
    )
    layer_input = BatchNormalization(axis=-1)(layer_input)
    layer_input = Concatenate()([layer_input, skip_input])
    layer_input = BatchNormalization(axis=-1)(layer_input)
    layer_input = LeakyReLU(NEGATIVE_RATIO)(layer_input)
    return layer_input
