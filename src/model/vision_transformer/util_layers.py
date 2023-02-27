
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.activations import gelu


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


class SpectralNormalizationConv(layers.Layer):
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
        # sigma.shape => [1, out_channel] @ [out_channel, N] => [1, N]
        sigma = tf.matmul(self.u, self.w, transpose_b=True)
        # sigma.shape => [1, N] @ [N, out_channel] => [1, out_channel]
        sigma = tf.reduce_sum(tf.matmul(sigma, self.u, transpose_b=True))
        self.layer.kernel = self.w / sigma
        return self.layer(x)


class SpectralNormalizationDense(layers.Layer):
    def __init__(self, layer, iteration=1):
        super(SpectralNormalization, self).__init__()
        self.iteration = iteration
        self.layer = layer

    def build(self, input_shape):
        # layers.kernel.shape => [in_channel, out_channel]
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=False)
        super(SpectralNormalization, self).build(input_shape)

    def call(self, x):
        self.w = self.layer.kernel
        # First: w_reshaped.shape => [in_channel, out_channel]
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
        sigma = tf.reduce_sum(
            tf.matmul(self.u, self.w, transpose_b=True) * self.u)
        self.layer.kernel = self.w / sigma
        return self.layer(x)
