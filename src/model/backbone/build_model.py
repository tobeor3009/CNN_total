# python baseModule
from copy import deepcopy
# tensorflow Module
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.python.keras.engine import training
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.layers import Concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

# user Module
from .layers import conv2d_bn, residual_block, residual_block_last, deconv2d, deconv2d_simple

KERNEL_SIZE = (3, 3)
WEIGHT_DECAY = 1e-2


def build_generator(
    input_image_shape,
    output_channels=4,
    depth=None,
    generator_power=32,
    activation="sigmoid"
):
    """U-Net Generator"""

    # Image input
    input_image = Input(shape=input_image_shape)

    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 3

    down_sample_layers = []

    fix_shape_layer_1 = conv2d_bn(
        x=input_image,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        strides=(1, 1),
        activation="leakyrelu"
    )
    fix_shape_layer_2 = residual_block(
        x=fix_shape_layer_1,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        downsample=False,
        activation="leakyrelu"
    )
    down_sample_layers.append((fix_shape_layer_1, fix_shape_layer_2))
    # Downsampling
    for depth_step in range(depth):
        down_sample_layer = residual_block(
            x=fix_shape_layer_2,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
            activation="leakyrelu"
        )
        fix_shape_layer_1 = residual_block(
            x=down_sample_layer,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
        fix_shape_layer_2 = residual_block(
            x=fix_shape_layer_1,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
        layer_collection = (down_sample_layer,
                            fix_shape_layer_1, fix_shape_layer_2)
        down_sample_layers.append(layer_collection)
    # upsampling
    for depth_step in range(depth, 0, -1):
        if depth_step == depth:
            fix_shape_layer_1 = deconv2d(
                down_sample_layers[depth_step][2],
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        else:
            fix_shape_layer_1 = deconv2d(
                upsampling_layer,
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        fix_shape_layer_2 = deconv2d(
            fix_shape_layer_1,
            down_sample_layers[depth_step][2],
            generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            upsample=False,
        )
        upsampling_layer = deconv2d(
            fix_shape_layer_2,
            down_sample_layers[depth_step - 1][0],
            generator_power * (2 ** (depth_step - 1)),
            kernel_size=KERNEL_SIZE,
            upsample=True,
            use_upsampling_layer=False,
        )
    # control output_channels
    output_layer = residual_block_last(
        x=upsampling_layer,
        filters=output_channels,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
        activation=activation,
        normalization=True
    )

    model = training.Model(input_image, output_layer, name='residual_unet')
    return model


def build_generator_stargan(
    input_image_shape,
    label_input,
    label_tensor,
    target_label_input,
    target_label_tensor,
    mode="add",
    output_channels=4,
    depth=None,
    generator_power=32,
    activation="sigmoid"
):
    """U-Net Generator"""

    # Image input
    input_image = Input(shape=input_image_shape)
    if mode == "add":
        input_image_with_label = input_image + label_tensor
    elif mode == "concatenate":
        input_image_with_label = keras_backend.concatenate([input_image, label_tensor], axis=-1)

    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 3

    down_sample_layers = []

    fix_shape_layer_1 = conv2d_bn(
        x=input_image_with_label,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        strides=(1, 1),
        activation="leakyrelu"
    )
    fix_shape_layer_2 = residual_block(
        x=fix_shape_layer_1,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        downsample=False,
        activation="leakyrelu"
    )
    down_sample_layers.append((fix_shape_layer_1, fix_shape_layer_2))
    # Downsampling
    for depth_step in range(depth):
        down_sample_layer = residual_block(
            x=fix_shape_layer_2,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
            activation="leakyrelu"
        )
        fix_shape_layer_1 = residual_block(
            x=down_sample_layer,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
        fix_shape_layer_2 = residual_block(
            x=fix_shape_layer_1,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
        layer_collection = (down_sample_layer,
                            fix_shape_layer_1, fix_shape_layer_2)
        down_sample_layers.append(layer_collection)

    if mode == "add":
        fix_shape_layer_2 = fix_shape_layer_2 + target_label_tensor
    elif mode == "concatenate":
        fix_shape_layer_2 = keras_backend.concatenate([fix_shape_layer_2, target_label_tensor], axis=-1)

    # upsampling
    for depth_step in range(depth, 0, -1):
        if depth_step == depth:
            fix_shape_layer_1 = deconv2d(
                down_sample_layers[depth_step][2],
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        else:
            fix_shape_layer_1 = deconv2d(
                upsampling_layer,
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        fix_shape_layer_2 = deconv2d(
            fix_shape_layer_1,
            down_sample_layers[depth_step][2],
            generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            upsample=False,
        )
        upsampling_layer = deconv2d(
            fix_shape_layer_2,
            down_sample_layers[depth_step - 1][0],
            generator_power * (2 ** (depth_step - 1)),
            kernel_size=KERNEL_SIZE,
            upsample=True,
            use_upsampling_layer=False,
        )
    # control output_channels
    output_layer = residual_block_last(
        x=upsampling_layer,
        filters=output_channels,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
        activation=activation,
        normalization=True
    )

    model = training.Model(
        [input_image, label_input, target_label_input], output_layer, name='residual_unet')
    return model


def build_generator_non_unet(
    input_image_shape,
    output_channels=4,
    depth=None,
    generator_power=32,
):
    # Image input
    input_image = Input(shape=input_image_shape)
    input_image_noised = GaussianNoise(0.1)(input_image)

    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 3

    fix_shape_layer_1 = conv2d_bn(
        x=input_image_noised,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        strides=(1, 1),
        activation="leakyrelu"
    )
    fix_shape_layer_2 = residual_block(
        x=fix_shape_layer_1,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        weight_decay=1e-4,
        downsample=False,
        activation="leakyrelu"
    )
    # Downsampling
    for depth_step in range(depth):
        down_sample_layer = residual_block(
            x=fix_shape_layer_2,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
            activation="leakyrelu"
        )
        fix_shape_layer_1 = residual_block(
            x=down_sample_layer,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
        fix_shape_layer_2 = residual_block(
            x=fix_shape_layer_1,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            weight_decay=1e-4,
            downsample=False,
            activation="leakyrelu"
        )
    # upsampling
    for depth_step in range(depth, 0, -1):
        if depth_step == depth:
            fix_shape_layer_1 = deconv2d_simple(
                fix_shape_layer_2,
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        else:
            fix_shape_layer_1 = deconv2d_simple(
                upsampling_layer,
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                upsample=False,
            )
        fix_shape_layer_2 = deconv2d_simple(
            fix_shape_layer_1,
            generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            upsample=False,
        )
        upsampling_layer = deconv2d_simple(
            fix_shape_layer_2,
            generator_power * (2 ** (depth_step - 1)),
            kernel_size=KERNEL_SIZE,
            upsample=True,
            use_upsampling_layer=False,
        )
    # control output_channels
    if output_channels is None:
        output_layer = upsampling_layer
    else:
        output_layer = residual_block_last(
            x=upsampling_layer,
            filters=output_channels,
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
            activation="sigmoid",
            normalization=True
        )
    return Model(input_image, output_layer)


def build_discriminator(
    input_image_shape,
    output_image_shape,
    depth=None,
    discriminator_power=32,
    num_class=1,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_image = Input(shape=input_image_shape)

    if output_image_shape is None:
        combined_images = original_image
    else:
        man_or_model_made_image = Input(shape=output_image_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_images = Concatenate(
            axis=-1)([original_image, man_or_model_made_image])

    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 3
    down_sampled_layer = conv2d_bn(
        x=combined_images,
        filters=discriminator_power,
        kernel_size=(3, 3),
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
            activation="leakyrelu"
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
            activation="leakyrelu"
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
            activation="leakyrelu"
        )

    validity = residual_block_last(
        x=down_sampled_layer,
        filters=num_class,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
        activation="sigmoid"
    )
    if output_image_shape is None:
        return Model(original_image, validity)
    else:
        return Model([original_image, man_or_model_made_image], validity)


def build_classifier(
    input_image_shape,
    classfier_power=32,
    depth=None,
    num_class=2,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    input_image = Input(shape=input_image_shape)
    dense_unit = 1024
    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 5
    down_sampled_layer = conv2d_bn(
        x=input_image,
        filters=classfier_power,
        kernel_size=(3, 3),
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
            activation="leakyrelu"
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
            activation="leakyrelu"
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
            activation="leakyrelu"
        )
    # (BATCH_SIZE, Filters)
    output = GlobalAveragePooling2D()(down_sampled_layer)
    # (BATCH_SIZE, 1024)
    output = Dense(units=dense_unit, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(units=dense_unit, activation="relu")(output)
    output = Dropout(0.5)(output)
    # (BATCH_SIZE, NUM_CLASS)
    output = Dense(units=num_class,
                   activation="sigmoid",
                   kernel_initializer="glorot_uniform")(output)
    return Model(input_image, output)


def build_ensemble_discriminator(
    input_image_shape,
    output_image_shape,
    depth=None,
    discriminator_power=32,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_image = Input(shape=input_image_shape)
    man_or_model_mad_image = Input(shape=output_image_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_images = Concatenate(
        axis=-1)([original_image, man_or_model_mad_image])

    if depth is None:
        image_size = input_image_shape[0]
        depth = 0
        while image_size != 1:
            image_size //= 2
            depth += 1
        depth -= 3

    # ----------------------------
    #  Define Filter Growing Layer
    # ----------------------------
    filter_growing_layer = conv2d_bn(
        x=combined_images,
        filters=discriminator_power,
        kernel_size=(3, 3),
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_growing_validity = residual_block_last(
        x=filter_growing_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )
    # ----------------------------
    #  Define Filter Shrinking Layer
    # ----------------------------
    filter_shrinking_layer = conv2d_bn(
        x=combined_images,
        filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
        kernel_size=(3, 3),
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth - 1, -1, -1):
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_shrinking_validity = residual_block_last(
        x=filter_shrinking_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )
    # ----------------------------
    #  Define Filter Fixed Layer
    # ----------------------------
    filter_fixed_layer = conv2d_bn(
        x=combined_images,
        filters=discriminator_power,
        kernel_size=(3, 3),
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_fixed_validity = residual_block_last(
        x=filter_fixed_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )
    validity = layers.Average()(
        [filter_growing_validity, filter_shrinking_validity, filter_fixed_validity]
    )
    return Model([original_image, man_or_model_mad_image], validity)
