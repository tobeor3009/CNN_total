import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, activations, Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.ops.gen_array_ops import size
from .coord import CoordinateChannel2D
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh


class HighwayMulti(layers.Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, dim, activation='relu', transform_gate_bias=-3, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dim = dim
        self.dense_1 = Dense(
            units=self.dim, bias_initializer=transform_gate_bias_initializer)

        super(HighwayMulti, self).__init__(**kwargs)

    def call(self, x, y):
        gate_input = layers.Concatenate(axis=-1)([x, y])
        transform_gate = self.dense_1(gate_input)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(self.dim,))(transform_gate)
        transformed_gated = Multiply()([transform_gate, x])
        identity_gated = Multiply()([carry_gate, y])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(HighwayMulti, self).get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config


class Highway(layers.Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, dim, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dim = dim
        self.dense_1 = Dense(
            units=self.dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_2 = Dense(units=self.dim)

        super(Highway, self).__init__(**kwargs)

    def call(self, x):
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(self.dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Highway, self).get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(tensor=input_tensor,
                      paddings=padding_tensor,
                      mode="REFLECT")


class UnsharpMasking2D(layers.Layer):
    def __init__(self, filters):
        super(UnsharpMasking2D, self).__init__()
        gauss_kernel_2d = get_gaussian_kernel(2, 0.0, 1.0)
        self.gauss_kernel = tf.tile(
            gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, filters, 1])

        self.pointwise_filter = tf.eye(filters, batch_shape=[1, 1])

    def call(self, input_tensor):
        blur_tensor = tf.nn.separable_conv2d(input_tensor,
                                             self.gauss_kernel,
                                             self.pointwise_filter,
                                             strides=[1, 1, 1, 1], padding='SAME')
        unsharp_mask_tensor = 2 * input_tensor - blur_tensor
        # because it used after tanh
        unsharp_mask_tensor = tf.clip_by_value(unsharp_mask_tensor, -1, 1)
        return unsharp_mask_tensor


def DecoderTransposeX2Block(filters):
    return layers.Conv2DTranspose(
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        use_bias=False,
    )


def get_input_label2image_tensor(label_len, target_shape,
                                 activation="tanh", negative_ratio=0.25,
                                 dropout_ratio=0.5, reduce_level=5):

    target_channel = target_shape[-1]
    reduce_size = (2 ** reduce_level)
    reduced_shape = (target_shape[0] // reduce_size,
                     target_shape[1] // reduce_size,
                     32)

    class_input = layers.Input(shape=(label_len,))
    class_tensor = layers.Dense(target_shape[0] * 2)(class_input)

    class_tensor = layers.Flatten()(class_tensor)
    class_tensor = layers.LeakyReLU(negative_ratio)(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Dense(np.prod(reduced_shape))(class_tensor)
    class_tensor = layers.LeakyReLU(negative_ratio)(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Reshape(reduced_shape)(class_tensor)
    for index in range(1, reduce_level):
        class_tensor = DecoderTransposeX2Block(32 * index)(class_tensor)
        class_tensor = layers.LeakyReLU(negative_ratio)(class_tensor)

    class_tensor = DecoderTransposeX2Block(target_channel)(class_tensor)
    class_tensor = layers.Reshape(target_shape)(class_tensor)
    class_tensor = Activation(activation)(class_tensor)

    return class_input, class_tensor


class HighwayResnetBlock(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super(HighwayResnetBlock, self).__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=1,
                                    padding="valid", kernel_initializer=kernel_init)
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.25)
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.use_highway is True:
            x = self.highway_layer(x, input_tensor)
        return x


class HighwayResnetEncoder(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super(HighwayResnetEncoder, self).__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=2,
                                    padding="valid", kernel_initializer=kernel_init)
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.25)
        if self.use_highway is True:
            self.pooling_layer = layers.AveragePooling2D(
                pool_size=2, strides=2, padding="same")
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.use_highway is True:
            source = self.pooling_layer(input_tensor)
            x = self.highway_layer(x, source)
        return x


class HighwayResnetDecoder(layers.Layer):
    def __init__(self, filters):
        super(HighwayResnetDecoder, self).__init__()
        self.unsharp_mask_layer = UnsharpMasking2D(filters)
        # Define Base Model Params
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()

        self.conv2d = layers.Conv2D(filters=filters * 4,
                                    kernel_size=(3, 3), strides=1,
                                    padding="valid", kernel_initializer=kernel_init)
        self.conv_after_pixel_shffle = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1,
                                                     padding="same", kernel_initializer=kernel_init)

        self.conv_before_upsample = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1,
                                                  padding="same", kernel_initializer=kernel_init)
        self.upsample_layer = layers.UpSampling2D(
            size=2, interpolation="bilinear")
        self.conv_after_upsample = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1,
                                                 padding="same", kernel_initializer=kernel_init)

        self.norm_layer = layers.BatchNormalization()
        # self.act_layer = layers.LeakyReLU(alpha=0.25)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        pixel_shuffle = self.padding_layer(input_tensor)
        pixel_shuffle = self.conv2d(pixel_shuffle)
        pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle, block_size=2)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        x = self.conv_before_upsample(input_tensor)
        x = self.upsample_layer(x)
        x = self.conv_after_upsample(x)

        output = self.highway_layer(pixel_shuffle, x)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        output = self.unsharp_mask_layer(output)
        return output


def get_highway_resnet_generator_unet(input_shape,
                                      init_filters, encoder_depth, middle_depth, last_channel_num,
                                      skip_connection=True):

    decoder_depth = encoder_depth
    kernel_init = RandomNormal(mean=0.0, stddev=0.02)

    middle_layers = []
    for middle_index in range(1, middle_depth + 1):
        if middle_index == 1:
            use_highway = False
        else:
            use_highway = True
        middle_layers.append(HighwayResnetBlock(
            init_filters * encoder_depth, use_highway=use_highway))
    middle_layers = Sequential(middle_layers)

    # model start
    input_tensor = layers.Input(shape=input_shape)
    coorded_tensor = CoordinateChannel2D(use_radius=True)(input_tensor)
    encoded_tensor = HighwayResnetBlock(
        init_filters, use_highway=False)(coorded_tensor)

    encoder_tensor_list = []
    for index in range(1, encoder_depth + 1):
        encoded_tensor = HighwayResnetBlock(
            init_filters * index, use_highway=False)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

        encoded_tensor = HighwayResnetBlock(
            init_filters * index)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

        encoded_tensor = HighwayResnetEncoder(
            init_filters * index)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

    decoded_tensor = middle_layers(encoded_tensor)

    for index in range(decoder_depth, 0, -1):

        decoded_tensor = HighwayResnetBlock(
            init_filters * index, use_highway=False)(decoded_tensor)
        decoded_tensor = HighwayResnetBlock(
            init_filters * index)(decoded_tensor)
        if skip_connection is True:
            target_layer_num = index * 3 - 1
            skip_connection_target = encoder_tensor_list[target_layer_num]
            decoded_tensor = layers.Concatenate(
                axis=-1)([decoded_tensor, skip_connection_target])
        decoded_tensor = HighwayResnetDecoder(
            init_filters * index)(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False)(decoded_tensor)

    last_modified_tensor = layers.Conv2D(filters=last_channel_num,
                                         kernel_size=(3, 3), strides=1,
                                         padding="same", kernel_initializer=kernel_init)(last_modified_tensor)
    last_modified_tensor = activations.tanh(last_modified_tensor)
    return Model(input_tensor, last_modified_tensor)


def get_highway_resnet_generator_stargan_unet(input_shape, label_len, target_label_len,
                                              init_filters, encoder_depth, middle_depth, last_channel_num,
                                              skip_connection=True):

    decoder_depth = encoder_depth
    kernel_init = RandomNormal(mean=0.0, stddev=0.02)

    input_label_shape = (input_shape[0] // (2 ** encoder_depth),
                         input_shape[1] // (2 ** encoder_depth),
                         init_filters * encoder_depth)

    target_label_shape = (input_shape[0] // (2 ** encoder_depth),
                          input_shape[1] // (2 ** encoder_depth),
                          init_filters * encoder_depth)
    class_input, class_tensor = get_input_label2image_tensor(label_len=label_len, target_shape=input_label_shape,
                                                             activation="tanh", negative_ratio=0.25,
                                                             dropout_ratio=0.5, reduce_level=5)
    target_class_input, target_class_tensor = get_input_label2image_tensor(label_len=target_label_len, target_shape=target_label_shape,
                                                                           activation="tanh", negative_ratio=0.25,
                                                                           dropout_ratio=0.5, reduce_level=5)
    middle_layers = []
    for middle_index in range(1, middle_depth + 1):
        if middle_index == 1:
            use_highway = False
        else:
            use_highway = True
        middle_layers.append(HighwayResnetBlock(
            init_filters * encoder_depth, use_highway=use_highway))
    middle_layers = Sequential(middle_layers)

    # model start
    input_tensor = layers.Input(shape=input_shape)
    coorded_tensor = CoordinateChannel2D(use_radius=True)(input_tensor)
    encoded_tensor = HighwayResnetBlock(
        init_filters, use_highway=False)(coorded_tensor)

    encoder_tensor_list = []
    for index in range(1, encoder_depth + 1):
        encoded_tensor = HighwayResnetBlock(
            init_filters * index, use_highway=False)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

        encoded_tensor = HighwayResnetBlock(
            init_filters * index)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

        encoded_tensor = HighwayResnetEncoder(
            init_filters * index)(encoded_tensor)
        encoder_tensor_list.append(encoded_tensor)

    encoded_tensor = layers.Concatenate(
        axis=-1)([encoded_tensor, class_tensor])

    feature_selcted_tensor = middle_layers(encoded_tensor)

    decoded_tensor = layers.Concatenate(
        axis=-1)([feature_selcted_tensor, target_class_tensor])
    for index in range(decoder_depth, 0, -1):

        decoded_tensor = HighwayResnetBlock(
            init_filters * index, use_highway=False)(decoded_tensor)
        decoded_tensor = HighwayResnetBlock(
            init_filters * index)(decoded_tensor)
        if skip_connection is True:
            target_layer_num = index * 3 - 1
            skip_connection_target = encoder_tensor_list[target_layer_num]
            decoded_tensor = layers.Concatenate(
                axis=-1)([decoded_tensor, skip_connection_target])
        decoded_tensor = HighwayResnetDecoder(
            init_filters * index)(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False)(decoded_tensor)

    last_modified_tensor = layers.Conv2D(filters=last_channel_num,
                                         kernel_size=(3, 3), strides=1,
                                         padding="same", kernel_initializer=kernel_init)(last_modified_tensor)
    last_modified_tensor = activations.tanh(last_modified_tensor)
    return Model([input_tensor, class_input, target_class_input], last_modified_tensor)


def get_discriminator(
    input_img_shape,
    output_img_shape,
    depth=None,
    init_filters=32,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_image = layers.Input(shape=input_img_shape)
    predicted_image = layers.Input(shape=output_img_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = layers.Concatenate(
        axis=-1)([original_image, predicted_image])

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3
    decoded_tensor = HighwayResnetBlock(
        init_filters, use_highway=False)(combined_imgs)
    for depth_step in range(depth):
        decoded_tensor = HighwayResnetBlock(
            init_filters * (2 ** ((depth_step + 1) // 2)), use_highway=False)(decoded_tensor)
        decoded_tensor = HighwayResnetBlock(
            init_filters * (2 ** ((depth_step + 1) // 2)), use_highway=True)(decoded_tensor)
        decoded_tensor = HighwayResnetBlock(
            init_filters * (2 ** ((depth_step + 1) // 2)), use_highway=False)(decoded_tensor)

    validity = HighwayResnetBlock(1, use_highway=False)(decoded_tensor)
    validity = activations.sigmoid(validity)
    return Model([original_image, predicted_image], validity)


def get_gaussian_kernel(size=2, mean=0.0, std=1.0):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.compat.v1.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)