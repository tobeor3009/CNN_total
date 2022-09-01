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
from tensorflow.nn import relu6
from tensorflow_addons.layers import InstanceNormalization

base_act = relu6

USE_CONV_BIAS = False
USE_DENSE_BIAS = False


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
        self.dense_1 = Dense(units=self.dim,
                             use_bias=USE_DENSE_BIAS, bias_initializer=transform_gate_bias_initializer)

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


class ConvBlock3D(layers.Layer):
    def __init__(self, filters, stride, use_act=True):
        super().__init__()
        self.use_act = use_act
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.conv3d = layers.Conv3D(filters=filters,
                                    kernel_size=3, strides=stride,
                                    padding="same", kernel_initializer=kernel_init,
                                    use_bias=USE_CONV_BIAS)
        self.norm_layer = layers.LayerNormalization(axis=-1)
        if self.use_act is True:
            self.act_layer = base_act

    def call(self, input_tensor):
        x = self.conv3d(input_tensor)
        x = self.norm_layer(x)
        if self.use_act is True:
            x = self.act_layer(x)
        return x


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs3D(layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 5
        ), f"Number of dimensions should be 4, got {len(input_shape)}"
        pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2], input_shape[3], 1)
            ),
            dtype="float32",
            trainable=True,
        )
        self.pe = tf.repeat(pe, repeats=[input_shape[4]], axis=-1)

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GCBlock(layers.Layer):
    def __init__(self, in_channel, ratio, fusion_types=('channel_add',), **kwargs):
        super().__init__(**kwargs)
        assert in_channel is not None, 'GCBlock needs in_channel'
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.ratio = ratio
        self.middle_channel = int(in_channel * ratio)
        self.fusion_types = fusion_types

        self.positional_emb = AddPositionEmbs3D()
        self.key_mask = layers.Conv3D(filters=in_channel,
                                      kernel_size=1,
                                      kernel_initializer=kaiming_initializer,
                                      padding="same",
                                      use_bias=USE_CONV_BIAS)
        self.value_mask = layers.Conv3D(filters=1,
                                        kernel_size=1,
                                        kernel_initializer=kaiming_initializer,
                                        padding="same",
                                        use_bias=USE_CONV_BIAS)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_mul_conv = None

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, x):

        x = self.positional_emb(x)

        # x.shape: [B, H, W, C]
        # key_mask.shape: [B, H, W, C]
        key_mask = self.key_mask(x)
        # key_mask.shape: [B, (H * W), C]
        key_mask = layers.Reshape((self.H * self.W, self.C))(key_mask)
        # key_mask.shape: [B, C, (H * W)]
        key_mask = layers.Permute((2, 1))(key_mask)

        # value_mask.shape: [B, H, W, 1]
        value_mask = self.value_mask(x)
        # value_mask.shape: [B, (H * W), 1]
        value_mask = layers.Reshape((self.H * self.W, 1))(value_mask)
        value_mask = self.softmax(value_mask)

        # [B, C, (H * W)] @ [B, (H * W), 1]
        # context_mask.shape: [B, C, 1]
        context_mask = tf.matmul(key_mask, value_mask)
        # context_mask.shape: [B, 1, 1, C]
        context_mask = layers.Reshape((1, 1, self.C))(context_mask)

        out = x
        if self.channel_mul_conv is not None:
            # [B, 1, 1, C]
            channel_mul_term = sigmoid(self.channel_mul_conv(context_mask))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [B, 1, 1, C]
            channel_add_term = self.channel_add_conv(context_mask)
            out = out + channel_add_term
        # out.shape: [B, H, W, C]
        return out


class HighwayResnetBlock3D(layers.Layer):
    def __init__(self, filters, use_highway=True, use_act=True):
        super().__init__()
        # Define Base Model Params
        self.use_highway = use_highway
        self.conv3d = ConvBlock3D(filters=filters,
                                  stride=1, use_act=use_act)
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        x = self.conv3d(input_tensor)
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
    def __init__(self, filters, strides):
        super().__init__()

        self.filters = filters
        self.conv_before_trans = layers.Conv3D(filters=filters,
                                               kernel_size=1, padding="same",
                                               strides=1, use_bias=USE_CONV_BIAS)
        self.conv_trans = layers.Conv3DTranspose(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=strides, use_bias=USE_CONV_BIAS)
        self.conv_after_trans = layers.Conv3D(filters=filters,
                                              kernel_size=1, padding="same",
                                              strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv3D(filters=filters,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)
        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        conv_trans = self.conv_before_trans(input_tensor)
        conv_trans = self.conv_trans(conv_trans)
        conv_trans = self.conv_after_trans(conv_trans)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(conv_trans, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class SkipUpsample3D(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.compress_block = Sequential([
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.LayerNormalization(axis=-1),
            layers.Activation("tanh")
        ])
        self.conv_block = Sequential([
            layers.Conv3D(filters, kernel_size=3, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.LayerNormalization(axis=-1),
            layers.Activation("tanh")
        ])

    def call(self, input_tensor, H):
        conv = self.compress_block(input_tensor)
        # shape: [B H W 1 C]
        conv = keras_backend.expand_dims(conv, axis=-2)
        conv = tf.repeat(conv, repeats=[H], axis=-2)
        conv = self.conv_block(conv)
        return conv


class DownBlock(layers.Layer):
    def __init__(self, filters, strides):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size=3, padding="same",
                                  strides=strides, use_bias=USE_CONV_BIAS)
        self.norm = InstanceNormalization()
        self.act = base_act

    def call(self, input_tensor):
        conv = self.conv(input_tensor)
        norm = self.norm(conv)
        act = self.act(norm)
        return act


class DenseBlock1D(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.dense = layers.Dense(filters, use_bias=USE_CONV_BIAS)
        self.norm = InstanceNormalization()
        self.act = base_act

    def call(self, input_tensor):
        dense = self.dense(input_tensor)
        norm = self.norm(dense)
        act = self.act(norm)
        return act


class DenseBlock2D(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.down_block = DownBlock(filters, strides=2)
        self.dense_block = DenseBlock1D(filters)
        self.compress_block = DownBlock(filters // 2, strides=1)

    def call(self, input_tensor):
        down_block = self.down_block(input_tensor)
        dense_block = self.dense_block(down_block)
        compress_block = self.compress_block(dense_block)

        return compress_block


class HighwayOutputLayer(layers.Layer):
    def __init__(self, last_channel_num, act="tanh", use_highway=True):
        super().__init__()
        self.use_highway = use_highway

        self.conv_1x1 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=1,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.conv_3x3 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=last_channel_num)
        self.act = layers.Activation(act)

    def call(self, input_tensor):
        conv_1x1 = self.conv_1x1(input_tensor)
        conv_3x3 = self.conv_3x3(input_tensor)
        output = self.highway_layer(conv_1x1, conv_3x3)
        output = self.act(output)

        return output
