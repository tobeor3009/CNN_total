from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, activations, Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu, softmax, sigmoid
from tensorflow_addons.activations import mish

base_act = gelu


class LayerArchive:
    def __init__(self):
        pass


class TensorArchive:
    def __init__(self):
        pass

# Reference: https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
# GCBlock Means Global Context Block


kaiming_initializer = tf.keras.initializers.HeNormal()


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
        transform_gate = self.dense_1(x)
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
    def __init__(self, filters, stride):
        super(ConvBlock, self).__init__()
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        self.padding_layer = ReflectionPadding2D()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=(3, 3), strides=stride,
                                    padding="valid", kernel_initializer=kernel_init)
        self.batch_norm = layers.BatchNormalization()
        self.act = base_act

    def call(self, input_tensor):
        x = self.padding_layer(input_tensor)
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x


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

        self.positional_emb = AddPositionEmbs2D()
        self.key_mask = layers.Conv2D(
            filters=in_channel, kernel_size=1, kernel_initializer=kaiming_initializer)
        self.value_mask = layers.Conv2D(
            filters=1, kernel_size=1, kernel_initializer=kaiming_initializer)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1)])
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


class HighwayResnetBlock(layers.Layer):
    def __init__(self, out_channel, in_channel=None,
                 include_context=True, context_ratio=3,
                 use_highway=True):
        super(HighwayResnetBlock, self).__init__()
        self.use_highway = use_highway
        self.include_context = include_context
        self.conv = ConvBlock(
            filters=out_channel, stride=1)
        if self.include_context is True:
            in_channel = out_channel if in_channel is None else in_channel
            self.gc_layer = GCBlock(
                in_channel=in_channel, ratio=context_ratio)
        if self.use_highway is True:
            self.highway_layer = HighwayMulti(dim=out_channel)
        self.norm_layer = layers.LayerNormalization(axis=-1)

    def call(self, input_tensor):
        x = input_tensor
        if self.include_context is True:
            x = self.gc_layer(x)
        x = self.conv(x)
        if self.use_highway is True:
            x = self.highway_layer(x, input_tensor)
            x = self.norm_layer(x)
        return x


class HighwayResnetEncoder(layers.Layer):
    def __init__(self, out_channel, in_channel=None, include_context=True, context_ratio=3, use_highway=True):
        super(HighwayResnetEncoder, self).__init__()
        self.include_context = include_context
        self.use_highway = use_highway

        self.conv = ConvBlock(
            filters=out_channel, stride=2)
        if self.include_context is True:
            in_channel = out_channel if in_channel is None else in_channel
            self.gc_layer = GCBlock(
                in_channel=in_channel, ratio=context_ratio)
        if self.use_highway is True:
            self.pooling_layer = layers.AveragePooling2D(
                pool_size=2, strides=2, padding="same")
            self.highway_layer = HighwayMulti(dim=out_channel)
        self.norm_layer = layers.LayerNormalization(axis=-1)

    def call(self, input_tensor):
        x = input_tensor
        if self.include_context is True:
            x = self.gc_layer(x)
        x = self.conv(x)
        if self.use_highway is True:
            source = self.pooling_layer(input_tensor)
            x = self.highway_layer(x, source)
            x = self.norm_layer(x)
        return x


class HighwayResnetDecoder(layers.Layer):
    def __init__(self, out_channel, in_channel=None, include_context=True, context_ratio=1, kernel_size=2, unsharp=False):
        super(HighwayResnetDecoder, self).__init__()

        self.include_context = include_context

        self.kernel_size = kernel_size
        self.unsharp = unsharp

        self.conv2d = HighwayResnetBlock(
            out_channel * (kernel_size ** 2), use_highway=False, include_context=False)
        self.conv_after_pixel_shffle = HighwayResnetBlock(
            out_channel, use_highway=False, include_context=False)

        self.conv_before_upsample = HighwayResnetBlock(
            out_channel, use_highway=False, include_context=False)
        self.upsample_layer = layers.UpSampling2D(
            size=kernel_size, interpolation="bilinear")
        self.conv_after_upsample = HighwayResnetBlock(
            out_channel, use_highway=False, include_context=False)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=out_channel)

        if self.include_context is True:
            in_channel = out_channel if in_channel is None else in_channel
            self.gc_layer = GCBlock(
                in_channel=in_channel, ratio=context_ratio)
        if self.unsharp is True:
            self.unsharp_mask_layer = UnsharpMasking2D(out_channel)

    def call(self, input_tensor):

        x = input_tensor
        if self.include_context is True:
            x = self.gc_layer(x)
        pixel_shuffle = self.conv2d(x)
        pixel_shuffle = tf.nn.depth_to_space(
            pixel_shuffle, block_size=self.kernel_size)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

        upsample = self.conv_before_upsample(x)
        upsample = self.upsample_layer(upsample)
        upsample = self.conv_after_upsample(upsample)

        output = self.highway_layer(pixel_shuffle, upsample)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        if self.unsharp is True:
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
