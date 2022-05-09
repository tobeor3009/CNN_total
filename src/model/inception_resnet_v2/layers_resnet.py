import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu, softmax, sigmoid
from .layers import UnsharpMasking2D
import math
USE_CONV_BIAS = True
USE_DENSE_BIAS = True
GC_BLOCK_RATIO = 0.125
kaiming_initializer = tf.keras.initializers.HeNormal()


def highway_conv2d(input_tensor, filters, kernel_size=(3, 3),
                   downsample=False, same_channel=True,
                   padding="same", activation="relu",
                   use_bias=False, groups=1, name=None):

    if groups == 1:
        def get_norm_layer(groups, name=None): return layers.BatchNormalization(axis=-1,
                                                                                scale=not use_bias,
                                                                                name=name)
    elif groups > 1:
        def get_norm_layer(groups, name=None): return GroupNormalization(groups=groups,
                                                                         axis=-1,
                                                                         scale=not use_bias,
                                                                         name=name)
    else:
        raise Exception(
            f"groups is int type variable, current state is group = {groups}")
    ################################################
    ################# Define Layer #################
    ################################################
    if downsample:
        strides = 2
    else:
        strides = 1
    if downsample or (not same_channel):
        residual_conv_layer = layers.Conv2D(filters,
                                            kernel_size=1,
                                            strides=strides,
                                            padding=padding,
                                            groups=groups,
                                            use_bias=use_bias)
        residual_norm_layer = get_norm_layer(groups)

    conv_layer = layers.Conv2D(filters,
                               kernel_size,
                               strides=strides,
                               padding=padding,
                               groups=groups,
                               use_bias=use_bias)
    norm_name = None if name is None else name + '_norm'
    norm_layer = get_norm_layer(groups, name=norm_name)

    act_name = None if name is None else name + '_act'
    if activation is None:
        pass
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6, name=act_name)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.3, name=act_name)
    else:
        act_layer = layers.Activation(activation, name=act_name)

    output_name = None if name is None else name + '_output'

    ################################################
    ################# Define call ##################
    ################################################
    if downsample or (not same_channel):
        residual = residual_conv_layer(input_tensor)
        residual = residual_norm_layer(residual)
        residual = act_layer(residual)
    else:
        residual = input_tensor

    conv = conv_layer(input_tensor)
    norm = norm_layer(conv)
    act = act_layer(norm)
    output = highway_multi(act, residual,
                           dim=filters, mode="2d", output_name=output_name)

    return output


def highway_decode2d(input_tensor, filters,
                     unsharp=False,
                     activation="relu",
                     use_bias=False, groups=1, name=None):
    if groups == 1:
        def get_norm_layer(groups, name=None): return layers.BatchNormalization(axis=-1,
                                                                                scale=not use_bias,
                                                                                name=name)
    elif groups >= 1:
        def get_norm_layer(groups, name=None): return GroupNormalization(groups=groups,
                                                                         axis=-1,
                                                                         scale=not use_bias,
                                                                         name=name)
    else:
        raise Exception(
            f"groups is int type variable, current state is group = {groups}")
    ################################################
    ################# Define Layer #################
    ################################################
    conv_before_pixel_shffle = layers.Conv2D(filters=filters,
                                             kernel_size=1, padding="same",
                                             strides=1, use_bias=USE_CONV_BIAS)
    conv_after_pixel_shffle = layers.Conv2D(filters=filters,
                                            kernel_size=1, padding="same",
                                            strides=1, use_bias=USE_CONV_BIAS)
    conv_before_upsample = layers.Conv2D(filters=filters,
                                         kernel_size=1, padding="same",
                                         strides=1, use_bias=USE_CONV_BIAS)
    upsample_layer = layers.UpSampling2D(size=2)
    conv_after_upsample = layers.Conv2D(filters=filters,
                                        kernel_size=1, padding="same",
                                        strides=1, use_bias=USE_CONV_BIAS)
    norm_layer = get_norm_layer(groups)
    if activation is None:
        pass
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.3)
    else:
        act_layer = layers.Activation(activation)
    output_name = None if name is None else name + '_output'
    if unsharp is True:
        unsharp_mask_layer = UnsharpMasking2D(filters)

    ################################################
    ################# Define call ##################
    ################################################
    pixel_shffle = conv_before_pixel_shffle(input_tensor)
    pixel_shffle = tf.nn.depth_to_space(pixel_shffle,
                                        block_size=2)
    pixel_shffle = conv_after_pixel_shffle(pixel_shffle)

    upsamle = conv_before_upsample(input_tensor)
    upsamle = upsample_layer(upsamle)
    upsamle = conv_after_upsample(upsamle)

    output = highway_multi(pixel_shffle, upsamle,
                           dim=filters, mode="2d", output_name=output_name)
    output = norm_layer(output)
    output = act_layer(output)

    if unsharp is True:
        output = unsharp_mask_layer(output)

    return output


def highway_multi(x, y, dim,
                  mode='3d', transform_gate_bias=-3, output_name=None):

    ################################################
    ################# Define Layer #################
    ################################################
    transform_gate_bias_initializer = Constant(transform_gate_bias)
    dense_1 = layers.Dense(units=dim,
                           use_bias=USE_DENSE_BIAS,
                           bias_initializer=transform_gate_bias_initializer)
    if mode == '2d':
        transform_gate = layers.GlobalAveragePooling2D()(x)
    elif mode == '3d':
        transform_gate = layers.GlobalAveragePooling3D()(x)
    ################################################
    ################# Define call ##################
    ################################################
    transform_gate = dense_1(transform_gate)
    transform_gate = layers.Activation("sigmoid")(transform_gate)
    carry_gate = layers.Lambda(lambda x: 1.0 - x,
                               output_shape=(dim,))(transform_gate)
    transformed_gated = layers.Multiply()([transform_gate, x])
    identity_gated = layers.Multiply()([carry_gate, y])
    value = layers.Add(name=output_name)([transformed_gated,
                                          identity_gated])
    return value


def vector_quantizer(latent_tensor, num_embeddings, embedding_dim, beta=0.25):
    ################################################
    ################# Define Layer #################
    ################################################
    beta = (beta)
    w_init = tf.random_uniform_initializer()
    w_init_value = w_init(shape=(embedding_dim, num_embeddings),
                          dtype="float32"
                          )
    embeddings = tf.Variable(initial_value=w_init_value,
                             trainable=True,
                             name="embeddings_vqvae"
                             )
    ################################################
    ################# Define call ##################
    ################################################
    # C = embedding_dim
    _, H, W, C = backend.int_shape()
    # latent_tensor: [B H W C]
    # flattend: [(B*H*W) C]
    flattend = backend.reshape(latent_tensor, [-1, embedding_dim])
    # encoding_indices: [N] with 0~num_embeddings-1
    encoding_indices = get_code_indices(flattend, embeddings)
    # encodings: [N num_embeddings]
    encodings = tf.one_hot(encoding_indices, num_embeddings)
    # quantized: [N, embedding_dim]
    quantized = tf.matmul(encodings, embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, [-1, H, W, C])


def get_code_indices(flattened_inputs, embeddings):
    # Calculate L2-normalized distance between the inputs and the codes.
    # flattend_inputs: [N embedding_dim]
    # embeddings: [embedding_dim num_embeddings]
    # similarity: [N num_embeddings]
    similarity = tf.matmul(flattened_inputs, embeddings)
    # [N 1] + [num_embeddings] - [N num_embeddings]
    # distances: [N num_embeddings]
    distances = (
        tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
        + tf.reduce_sum(embeddings ** 2, axis=0)
        - 2 * similarity
    )

    # Derive the indices for minimum distances.
    # encoding_indices: [N]
    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            # This parameter is best kept between [0.25, 2] as per the paper.
            beta
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
