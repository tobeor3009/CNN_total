
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.activations import gelu
DEFAULT_SN_ITER = 7


def get_norm_layer(norm, axis=-1, name=None):
    if norm == "layer":
        norm_layer = layers.LayerNormalization(epsilon=1e-5,
                                               axis=axis,
                                               name=name)
    elif norm == "batch":
        norm_layer = layers.BatchNormalization(axis=axis,
                                               scale=False,
                                               name=name)
    elif norm == "instance":
        norm_layer = InstanceNormalization(axis=axis,
                                           name=name)
    elif norm is None:
        def norm_layer(x): return x
    return norm_layer


def get_act_layer(activation, name=None):
    if activation is None:
        def act_layer(x): return x
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6, name=name)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.2, name=name)
    elif activation == "gelu":
        act_layer = gelu
    else:
        act_layer = layers.Activation(activation, name=name)
    return act_layer


def drop_path_(inputs, drop_prob, is_training):

    # Bypass in non-training mode
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    input_shape = tf.shape(inputs)
    batch_num = input_shape[0]
    rank = len(input_shape)

    shape = (batch_num,) + (1,) * (rank - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    path_mask = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * path_mask
    return output


@tf.function
def spectral_norm(layer_kernel, kernel_shape, iteration):
    with tf.init_scope():
        # First: kernel_reshaped.shape => [N, out_channel]
        # After: kernel_reshaped.shape => [1, out_channel]
        print(layer_kernel, kernel_shape)
        kernel_reshaped = tf.reshape(layer_kernel,
                                     [-1, kernel_shape[-1]])
        u = tf.ones([1, kernel_shape[-1]])
        for _ in range(iteration):
            # First: u_hat.shape => [N, out_channel] @ [out_channel, 1] => [N, 1]
            # After: u_hat.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
            u_hat = tf.matmul(kernel_reshaped, u, transpose_b=True)
            u_hat_norm = tf.norm(u_hat)
            # First: v.shape => [N, 1]
            # After: v.shape => [1, 1]
            v = u_hat / u_hat_norm
            # First: v_hat.shape => [out_channel, N] @ [N, 1] => [out_channel, 1]
            # After: v_hat.shape => [out_channel, 1] @ [1, 1] => [out_channel, 1]
            v_hat = tf.matmul(kernel_reshaped, v, transpose_a=True)
            u = tf.transpose(v_hat)
            # First: kernel_reshaped.shape => [1, N] @ [N, out_channel] => [1, out_channel]
            # After: kernel_reshaped.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
            kernel_reshaped = tf.matmul(v, kernel_reshaped, transpose_a=True)
        # kernel_norm.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
        kernel_norm = tf.matmul(u, kernel_reshaped, transpose_b=True)
        # kernel_norm.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
        kernel_norm = tf.matmul(kernel_norm, u)
        # kernel_norm.shape => [] (scalar)
        kernel_norm = tf.reduce_sum(kernel_norm)
        return kernel_norm


class drop_path(layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path_(x, self.drop_prob, training)


class DenseLayer(layers.Layer):
    def __init__(self, units, activation=None, use_sn=False,
                 iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_sn = use_sn
        if "use_bias" in kwargs:
            use_bias = kwargs["use_bias"]
            del kwargs["use_bias"]
        else:
            use_bias = True
        self.use_bias = False if use_sn else use_bias
        self.iteration = iteration

        # Define layer
        def get_dense_fn():
            return layers.Dense(self.units,
                                use_bias=self.use_bias,
                                **kwargs)
        if use_sn:
            self.dense = SpectralNormalization(get_dense_fn(),
                                               power_iterations=iteration)
        else:
            self.dense = get_dense_fn()
        self.activation_layer = get_act_layer(activation)

    def build(self, input_shape):
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        outputs = self.dense(inputs)
        outputs = self.activation_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class Conv1DLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None,
                 use_sn=False, iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_sn = use_sn
        if "use_bias" in kwargs:
            use_bias = kwargs["use_bias"]
            del kwargs["use_bias"]
        else:
            use_bias = True
        self.use_bias = False if use_sn else use_bias
        self.iteration = iteration
        # Define Layer

        def get_conv1d_fn():
            return layers.Conv1D(self.filters, kernel_size=self.kernel_size,
                                 strides=self.strides, padding=self.padding, use_bias=self.use_bias,
                                 **kwargs)
        if use_sn:
            self.conv1d = SpectralNormalization(get_conv1d_fn(),
                                                power_iterations=iteration)
        else:
            self.conv1d = get_conv1d_fn()
        self.activation_layer = get_act_layer(activation)

    def build(self, input_shape):
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.activation_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv1d.compute_output_shape(input_shape)


class Conv2DLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None,
                 use_sn=False, iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_sn = use_sn
        if "use_bias" in kwargs:
            use_bias = kwargs["use_bias"]
            del kwargs["use_bias"]
        else:
            use_bias = True
        self.use_bias = False if use_sn else use_bias
        self.iteration = iteration
        # Define Layer

        def get_conv2d_fn():
            return layers.Conv2D(self.filters, kernel_size=self.kernel_size,
                                 strides=self.strides, padding=self.padding, use_bias=self.use_bias,
                                 **kwargs)
        if use_sn:
            self.conv2d = SpectralNormalization(get_conv2d_fn(),
                                                power_iterations=iteration)
        else:
            self.conv2d = get_conv2d_fn()
        self.activation_layer = get_act_layer(activation)

    def build(self, input_shape):
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        outputs = self.conv2d(inputs)
        outputs = self.activation_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv2d.compute_output_shape(input_shape)


class Conv3DLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid', activation=None,
                 use_sn=False, iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_sn = use_sn
        if "use_bias" in kwargs:
            use_bias = kwargs["use_bias"]
            del kwargs["use_bias"]
        else:
            use_bias = True
        self.use_bias = False if use_sn else use_bias

        self.iteration = iteration
        # Define Layer

        def get_conv3d_fn():
            return layers.Conv3D(self.filters, kernel_size=self.kernel_size,
                                 strides=self.strides, padding=self.padding, use_bias=self.use_bias,
                                 **kwargs)
        if use_sn:
            self.conv3d = SpectralNormalization(get_conv3d_fn(),
                                                power_iterations=iteration)
        else:
            self.conv3d = get_conv3d_fn()
        self.activation_layer = get_act_layer(activation)

    def build(self, input_shape):
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        outputs = self.conv3d(inputs)
        outputs = self.activation_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv3d.compute_output_shape(input_shape)


# class SpectralNormalization(layers.Wrapper):
#     def __init__(self, layer, iteration=1, **kwargs):
#         super(SpectralNormalization, self).__init__(layer, **kwargs)
#         self.iteration = iteration
#         self.layer = layer

#     def build(self, input_shape):
#         # super(SpectralNormalization, self).build(input_shape)
#         self.layer.build(input_shape)
#         kernel_shape = self.layer.kernel.shape.as_list()
#         # u.shape = [1, out_channel]
#         self.u = self.add_weight(shape=(1, kernel_shape[-1]), initializer="glorot_uniform",
#                                  trainable=False, name="sn_u")

#     def call(self, x):
#         # N = kernel_w * kernel_h * in_channel
#         layer_kernel = self.layer.kernel
#         kernel_norm, u = spectral_norm(layer_kernel, self.u,
#                                        self.iteration)
#         self.u.assign(u)
#         self.layer.kernel = layer_kernel / kernel_norm
#         return self.layer(x)
