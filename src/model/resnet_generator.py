import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, activations, Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from .coord import CoordinateChannel2D


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

    def call(self, input_tensor, mask=None):
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


# class UnsharpMasking2D(layers.Layer):
#     def __init__(self):
#         super(UnsharpMasking2D, self).__init__()
#         self.unsharp_kernel = keras_backend.constant([[0, -1, 0],
#                                                       [-1, 5, 1],
#                                                       [0, -1, 0]])

#     def call(self, input_tensor, mask=None):


class HighWayResnetBlock(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super(HighWayResnetBlock, self).__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=1,
                                    padding="valid", kernel_initializer=kernel_init)
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.25)
        initializer = tf.zeros_initializer()
        # if self.use_highway is True:
        #     self.highway_coef = tf.Variable(
        #         initial_value=initializer(shape=(1,)), trainable=True)

    def call(self, input_tensor, mask=None):

        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        # if self.use_highway is True:
        #     return self.highway_coef * x + (1 - self.highway_coef) * input_tensor
        # else:
        return x


class HighWayResnetEncoder(layers.Layer):
    def __init__(self, filters):
        super(HighWayResnetEncoder, self).__init__()
        # Define Base Model Params
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.pooling_layer = layers.AveragePooling2D(pool_size=(2, 2))
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=2,
                                    padding="valid", kernel_initializer=kernel_init)
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.25)
        initializer = tf.zeros_initializer()
        # self.highway_coef = tf.Variable(
        #     initial_value=initializer(shape=(1,)), trainable=True)

    def call(self, input_tensor, mask=None):

        # source = self.pooling_layer(input_tensor)
        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
        # return self.highway_coef * x + (1 - self.highway_coef) * source


class HighWayResnetDecoder(layers.Layer):
    def __init__(self, filters):
        super(HighWayResnetDecoder, self).__init__()
        # Define Base Model Params
        # kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        # self.conv2d = layers.Conv2DTranspose(filters=filters,
        #                                      kernel_size=(3, 3), strides=2,
        #                                      padding="same", kernel_initializer=kernel_init)
        # self.norm = layers.BatchNormalization()
        # self.activation = layers.LeakyReLU(alpha=0.25)
        # initializer = tf.zeros_initializer()
        # self.highway_coef = tf.Variable(
        #     initial_value=initializer(shape=(1,)), trainable=True)

    def call(self, input_tensor, mask=None):

        pixel_shuffle = tf.nn.depth_to_space(input_tensor, block_size=2)
        # x = self.conv2d(input_tensor)
        # x = self.norm(x)
        # x = self.activation(x)

        return pixel_shuffle
        # return self.highway_coef * x + (1 - self.highway_coef) * pixel_shuffle


def get_highway_resnet_generator(input_shape, init_filters, encoder_depth, middle_depth, last_channel_num):

    decoder_depth = encoder_depth
    kernel_init = RandomNormal(mean=0.0, stddev=0.02)

    encoder_layers = []
    for index in range(1, encoder_depth + 1):
        encoder_layers.append(HighWayResnetBlock(
            init_filters * index, use_highway=False))
        encoder_layers.append(HighWayResnetBlock(init_filters * index))
        encoder_layers.append(HighWayResnetEncoder(init_filters * index))
    encoder_layers = Sequential(encoder_layers)

    middle_layers = []

    for _ in range(1, middle_depth + 1):
        middle_layers.append(HighWayResnetBlock(init_filters * index))
    middle_layers = Sequential(middle_layers)

    decoder_layers = []
    for index in range(decoder_depth, 0, -1):
        decoder_layers.append(HighWayResnetBlock(
            init_filters * index * 4, use_highway=False))
        decoder_layers.append(HighWayResnetBlock(init_filters * index * 4))
        decoder_layers.append(HighWayResnetDecoder(init_filters * index))
    decoder_layers = Sequential(decoder_layers)

    # model start
    input_tensor = layers.Input(shape=input_shape)
    coorded_tensor = CoordinateChannel2D(use_radius=True)(input_tensor)
    conved_tensor = HighWayResnetBlock(
        init_filters, use_highway=False)(coorded_tensor)

    encoded_tensor = encoder_layers(conved_tensor)
    feature_selcted_tensor = middle_layers(encoded_tensor)
    decoded_tensor = decoder_layers(feature_selcted_tensor)
    last_modified_tensor = layers.Conv2D(filters=last_channel_num,
                                         kernel_size=(3, 3), strides=1,
                                         padding="same", kernel_initializer=kernel_init)(decoded_tensor)
    last_modified_tensor = activations.tanh(last_modified_tensor)
    return Model(input_tensor, last_modified_tensor)
