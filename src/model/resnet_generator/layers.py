import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, activations, Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu
from tensorflow_addons.activations import mish

base_act = gelu


class LayerArchive:
    def __init__(self):
        pass


class TensorArchive:
    def __init__(self):
        pass


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
        transform_gate = layers.GlobalAveragePooling2D()(x)
        transform_gate = self.dense_1(transform_gate)
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


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs2D(layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 4
        ), f"Number of dimensions should be 4, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2], 1)
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
                     target_shape[2])
    class_input = layers.Input(shape=(label_len,))
    class_tensor = layers.Dense(np.prod(reduced_shape) // 2)(class_input)
    class_tensor = base_act(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Dense(np.prod(reduced_shape))(class_tensor)
    class_tensor = base_act(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Reshape(reduced_shape)(class_tensor)
    for index in range(1, reduce_level):
        class_tensor = DecoderTransposeX2Block(
            target_shape[2] * index)(class_tensor)
        class_tensor = Activation(activation)(class_tensor)
    class_tensor = DecoderTransposeX2Block(target_channel)(class_tensor)
    class_tensor = layers.Reshape(target_shape)(class_tensor)
    class_tensor = Activation(activation)(class_tensor)

    return class_input, class_tensor


class ConvBlock(layers.Layer):
    def __init__(self, filters, stride, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=stride,
                                    padding="valid", kernel_initializer=kernel_init)
        self.norm_layer = layers.LayerNormalization(axis=-1)
        if self.use_act is True:
            self.act_layer = base_act

    def call(self, input_tensor):
        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.norm_layer(x)
        if self.use_act is True:
            x = self.act_layer(x)
        return x


class HighwayResnetBlock(layers.Layer):
    def __init__(self, out_channel, use_highway=True, use_act=False):
        super(HighwayResnetBlock, self).__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        self.use_act = use_act
        self.conv = ConvBlock(
            filters=out_channel, stride=1, use_act=use_act)
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=out_channel)
            self.norm_layer = layers.LayerNormalization(axis=-1)
            if self.use_act is True:
                self.act_layer = base_act

    def call(self, input_tensor):

        x = self.conv(input_tensor)
        if self.use_highway is True:
            x = self.highway_layer(x, input_tensor)
            x = self.norm_layer(x)
            if self.use_act is True:
                x = self.act_layer(x)
        return x


class HighwayResnetEncoder(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super(HighwayResnetEncoder, self).__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        self.depthwise_separable_conv = ConvBlock(
            filters=filters, stride=2)
        if self.use_highway is True:
            self.pooling_layer = layers.AveragePooling2D(
                pool_size=2, strides=2, padding="same")
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.depthwise_separable_conv(input_tensor)
        if self.use_highway is True:
            source = self.pooling_layer(input_tensor)
            x = self.highway_layer(x, source)
        return x


class HighwayResnetDecoder(layers.Layer):
    def __init__(self, filters, kernel_size=2, unsharp=False):
        super(HighwayResnetDecoder, self).__init__()

        self.kernel_size = kernel_size
        self.unsharp = unsharp
        self.unsharp_mask_layer = UnsharpMasking2D(filters)

        self.conv2d = HighwayResnetBlock(
            filters * (kernel_size ** 2), use_highway=False)
        self.conv_after_pixel_shffle = HighwayResnetBlock(
            filters, use_highway=False)

        self.conv_before_upsample = HighwayResnetBlock(
            filters, use_highway=False)
        self.upsample_layer = layers.UpSampling2D(
            size=kernel_size, interpolation="bilinear")
        self.conv_after_upsample = HighwayResnetBlock(
            filters, use_highway=False)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        pixel_shuffle = self.conv2d(input_tensor)
        pixel_shuffle = tf.nn.depth_to_space(
            pixel_shuffle, block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        x = self.conv_before_upsample(input_tensor)
        x = self.upsample_layer(x)
        x = self.conv_after_upsample(x)

        output = self.highway_layer(pixel_shuffle, x)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        if self.unsharp:
            output = self.unsharp_mask_layer(output)
        return output


def get_gaussian_kernel(size=2, mean=0.0, std=1.0):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.compat.v1.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)
