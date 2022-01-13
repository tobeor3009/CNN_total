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

        transform_gate = layers.GlobalAveragePooling3D()(x)
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


def DecoderTransposeX2Block3D(filters):
    return layers.Conv3DTranspose(
        filters,
        kernel_size=3,
        strides=2,
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
        class_tensor = DecoderTransposeX2Block3D(
            target_shape[2] * index)(class_tensor)
        class_tensor = Activation(activation)(class_tensor)
    class_tensor = DecoderTransposeX2Block3D(target_channel)(class_tensor)
    class_tensor = layers.Reshape(target_shape)(class_tensor)
    class_tensor = Activation(activation)(class_tensor)

    return class_input, class_tensor


class ConvBlock3D(layers.Layer):
    def __init__(self, filters, stride):
        super().__init__()
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.conv3d = layers.Conv3D(filters=filters,
                                    kernel_size=3, strides=stride,
                                    padding="same", kernel_initializer=kernel_init)
        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = base_act

    def call(self, input_tensor):
        x = self.conv3d(input_tensor)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        return x


class HighwayResnetBlock3D(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super().__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        self.depthwise_separable_conv = ConvBlock3D(
            filters=filters, stride=1)
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.depthwise_separable_conv(input_tensor)
        if self.use_highway is True:
            x = self.highway_layer(x, input_tensor)
        return x


class HighwayResnetEncoder3D(layers.Layer):
    def __init__(self, filters, use_highway=True):
        super().__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        self.conv3d = ConvBlock3D(
            filters=filters, stride=2)
        if self.use_highway is True:
            self.pooling_layer = layers.AveragePooling3D(
                pool_size=2, strides=2, padding="same")
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.conv3d(input_tensor)
        if self.use_highway is True:
            source = self.pooling_layer(input_tensor)
            x = self.highway_layer(x, source)
        return x


class HighwayResnetDecoder3D(layers.Layer):
    def __init__(self, filters, kernel_size=2):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.conv3d = HighwayResnetBlock3D(
            filters * (kernel_size ** 3), use_highway=False)
        self.conv_after_pixel_shffle = HighwayResnetBlock3D(
            filters, use_highway=False)

        self.conv_before_upsample = HighwayResnetBlock3D(
            filters, use_highway=False)
        self.upsample_layer = layers.UpSampling3D(size=kernel_size)
        self.conv_after_upsample = HighwayResnetBlock3D(
            filters, use_highway=False)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters)

    def build(self, input_shape):
        _, self.H, self.W, self.T, self.C = input_shape
        self.new_H = self.H * self.kernel_size
        self.new_W = self.W * self.kernel_size
        self.new_T = self.T * self.kernel_size

    def call(self, input_tensor):

        pixel_shuffle = self.conv3d(input_tensor)
        pixel_shuffle = layers.Reshape(
            (self.H, self.W, self.T * (self.kernel_size ** 3) * self.filters))(pixel_shuffle)
        pixel_shuffle = tf.nn.depth_to_space(
            pixel_shuffle, block_size=self.kernel_size)
        pixel_shuffle = layers.Reshape(
            (self.new_H, self.new_W, self.new_T, self.filters))(pixel_shuffle)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(pixel_shuffle, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output
