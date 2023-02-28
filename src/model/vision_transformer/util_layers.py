
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.activations import gelu
DEFAULT_SN_ITER = 3


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
        act_layer = layers.LeakyReLU(0.3, name=name)
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


class drop_path(layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path_(x, self.drop_prob, training)


class DenseLayer(layers.Layer):
    def __init__(self, units, activation=None, use_sn=False,
                 iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.dense = layers.Dense(self.units, self.use_bias, **kwargs)
        self.activation_layer = get_act_layer(activation)

    def build(self, input_shape):
        if self.use_sn:
            self.dense_sn = SpectralNormalization(self.dense,
                                                  iteration=self.iteration)
        super().build(input_shape)

    def call(self, inputs):
        if self.use_sn:
            outputs = self.dense_sn(inputs)
        else:
            outputs = self.dense(inputs)
        outputs = self.activation_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        config = {'units': self.units, 'activation': self.activation,
                  'use_sn': self.use_sn, 'iteration': self.iteration}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Conv2DLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None,
                 use_sn=False, iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.conv2d = layers.Conv2D(self.filters, self.kernel_size,
                                    strides=self.strides, padding=self.padding, use_bias=self.use_bias,
                                    **kwargs)

    def build(self, input_shape):
        if self.use_sn:
            self.conv2d_sn = SpectralNormalization(self.conv2d,
                                                   iteration=self.iteration)
        super().build(input_shape)

    def call(self, inputs):
        if self.use_sn:
            outputs = self.conv2d_sn(inputs)
        else:
            outputs = self.conv2d(inputs)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv2d.compute_output_shape(input_shape)

    def get_config(self):
        config = {'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
                  'padding': self.padding, 'activation': self.activation, 'use_sn': self.use_sn, 'iteration': self.iteration}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Conv3DLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid', activation=None,
                 use_sn=False, iteration=DEFAULT_SN_ITER, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.conv3d = layers.Conv3D(self.filters, self.kernel_size,
                                    strides=self.strides, padding=self.padding, use_bias=self.use_bias,
                                    **kwargs)

    def build(self, input_shape):
        if self.use_sn:
            self.conv3d_sn = SpectralNormalization(self.conv3d,
                                                   iteration=self.iteration)
        super().build(input_shape)

    def call(self, inputs):
        if self.use_sn:
            outputs = self.conv3d_sn(inputs)
        else:
            outputs = self.conv3d(inputs)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv3d.compute_output_shape(input_shape)

    def get_config(self):
        config = {'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
                  'padding': self.padding, 'activation': self.activation, 'use_sn': self.use_sn, 'iteration': self.iteration}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# in can wrap conv or dense layer


class SpectralNormalization(layers.Layer):
    def __init__(self, layer, iteration=1):
        super(SpectralNormalization, self).__init__()
        self.iteration = iteration
        self.layer = layer

    def build(self, input_shape):
        # layers.kernel.shape => [kernel_w, kernel_h, in_channel, out_channel]
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        # u.shape = [1, out_channel]
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=False)
        super(SpectralNormalization, self).build(input_shape)

    def call(self, x):
        # N = kernel_w * kernel_h * in_channel
        self.w = self.layer.kernel
        # First: w_reshaped.shape => [N, out_channel]
        # After: w_reshaped.shape => [1, out_channel]
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        for _ in range(self.iteration):
            # First: u_hat.shape => [N, out_channel] @ [out_channel, 1] => [N, 1]
            # After: u_hat.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
            u_hat = tf.matmul(w_reshaped, self.u, transpose_b=True)
            u_hat_norm = tf.norm(u_hat)
            # First: v.shape => [N, 1]
            # After: v.shape => [1, 1]
            v = u_hat / u_hat_norm
            # First: v_hat.shape => [out_channel, N] @ [N, 1] => [out_channel, 1]
            # After: v_hat.shape => [out_channel, 1] @ [1, 1] => [out_channel, 1]
            v_hat = tf.matmul(w_reshaped, v, transpose_a=True)
            self.u.assign(v_hat)
            # First: w_reshaped.shape => [1, N] @ [N, out_channel] => [1, out_channel]
            # After: w_reshaped.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
            w_reshaped = tf.matmul(v, w_reshaped, transpose_a=True)
        # sigma.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
        sigma = tf.matmul(self.u, w_reshaped, transpose_b=True)
        # sigma.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
        sigma = tf.matmul(sigma, self.u)
        # sigma.shape => [] (scalar)
        sigma = tf.reduce_sum(sigma)
        self.layer.kernel = self.w / sigma
        return self.layer(x)


class SpectralNormalizationConv(layers.Layer):
    def __init__(self, layer, iteration=1):
        super().__init__()
        self.iteration = iteration
        self.layer = layer

    def build(self, input_shape):
        # layers.kernel.shape => [kernel_w, kernel_h, in_channel, out_channel]
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        # u.shape = [1, out_channel]
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=False)
        super().build(input_shape)

    def call(self, x):
        # N = kernel_w * kernel_h * in_channel
        self.w = self.layer.kernel
        # First: w_reshaped.shape => [N, out_channel]
        # After: w_reshaped.shape => [1, out_channel]
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        for _ in range(self.iteration):
            # First: u_hat.shape => [N, out_channel] @ [out_channel, 1] => [N, 1]
            # After: u_hat.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
            u_hat = tf.matmul(w_reshaped, self.u, transpose_b=True)
            u_hat_norm = tf.norm(u_hat)
            # First: v.shape => [N, 1]
            # After: v.shape => [1, 1]
            v = u_hat / u_hat_norm
            # First: v_hat.shape => [out_channel, N] @ [N, 1] => [out_channel, 1]
            # After: v_hat.shape => [out_channel, 1] @ [1, 1] => [out_channel, 1]
            v_hat = tf.matmul(w_reshaped, v, transpose_a=True)
            self.u.assign(v_hat)
            # First: w_reshaped.shape => [1, N] @ [N, out_channel] => [1, out_channel]
            # After: w_reshaped.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
            w_reshaped = tf.matmul(v, w_reshaped, transpose_a=True)
        # sigma.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
        sigma = tf.matmul(self.u, w_reshaped, transpose_b=True)
        # sigma.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
        sigma = tf.matmul(sigma, self.u)
        # sigma.shape => [] (scalar)
        sigma = tf.reduce_sum(sigma)
        self.layer.kernel = self.w / sigma
        return self.layer(x)


class SpectralNormalizationDense(layers.Layer):
    def __init__(self, layer, iteration=1):
        super().__init__()
        self.iteration = iteration
        self.layer = layer

    def build(self, input_shape):
        # layers.kernel.shape => [in_channel, out_channel]
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        # u.shape => [1, out_channel]
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=False)
        super().build(input_shape)

    def call(self, x):
        self.w = self.layer.kernel
        # First: w_reshaped.shape => [in_channel, out_channel]
        # After: w_reshaped.shape => [1, out_channel]
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        for _ in range(self.iteration):
            # First: u_hat.shape => [in_channel, out_channel] @ [out_channel, 1] => [in_channel, 1]
            # After: u_hat.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
            u_hat = tf.matmul(w_reshaped, self.u, transpose_b=True)
            u_hat_norm = tf.norm(u_hat)
            # First: v.shape => [in_channel, 1]
            # After: v.shape => [1, 1]
            v = u_hat / u_hat_norm
            # First: v_hat.shape => [out_channel, in_channel] @ [in_channel, 1] => [out_channel, 1]
            # After: v_hat.shape => [out_channel, 1] @ [1, 1] => [out_channel, 1]
            v_hat = tf.matmul(w_reshaped, v, transpose_a=True)
            self.u.assign(v_hat)
            # First: w_reshaped.shape => [1, in_channel] @ [in_channel, out_channel] => [1, out_channel]
            # After: w_reshaped.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
            w_reshaped = tf.matmul(v, w_reshaped, transpose_a=True)

        # sigma.shape => [1, out_channel] @ [out_channel, 1] => [1, 1]
        sigma = tf.matmul(self.u, w_reshaped, transpose_b=True)
        # sigma.shape => [1, 1] @ [1, out_channel] => [1, out_channel]
        sigma = tf.matmul(sigma, self.u)
        # sigma.shape => [] (scalar)
        sigma = tf.reduce_sum(sigma)
        self.layer.kernel = self.w / sigma
        return self.layer(x)
