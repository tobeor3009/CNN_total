import tensorflow as tf
from .base_model_resnet import HighWayResnet2D, HighWayDecoder2D, highway_conv2d
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend


def get_multi_scale_task_model(input_shape, num_class, block_size=16,
                               encoder_output_filter=16,
                               groups=10, num_downsample=5,
                               base_act="relu", last_act="tanh",
                               latent_dim=16):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    _, H, W, C = encoder.output.shape

    mean_std_layer = MeanSTD(latent_dim=latent_dim, name="mean_var")
    sampling_layer = Sampling(name="sampling")
    decode_dense_layer = layers.Dense(H * W * C // 64,
                                      activation=tf.nn.relu6)
    ################################################
    ################# Define call ##################
    ################################################
    input_tensor = encoder.input
    encoder_output = encoder(input_tensor)
    x = highway_conv2d(input_tensor=encoder.output, filters=C // 2,
                       downsample=True, same_channel=False,
                       padding=padding, activation=base_act,
                       groups=1)
    x = highway_conv2d(input_tensor=x, filters=C // 4,
                       downsample=True, same_channel=False,
                       padding=padding, activation=base_act,
                       groups=1)
    z_mean, z_log_var = mean_std_layer(x)
    sampling_z = sampling_layer([z_mean, z_log_var])
    z = decode_dense_layer(sampling_z)
    z = backend.reshape(z, (-1, H // 4, W // 4, C // 4))
    recon_output = HighWayDecoder2D(input_tensor=z,
                                    encoder=None, skip_connection_layer_names=None,
                                    last_filter=input_shape[-1],
                                    block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                    base_act=base_act, last_act=last_act, name_prefix="recon")

    seg_output = HighWayDecoder2D(input_tensor=encoder_output, encoder=encoder,
                                  skip_connection_layer_names=SKIP_CONNECTION_LAYER_NAMES,
                                  last_filter=num_class,
                                  block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act="sigmoid", name_prefix="seg")

    classification_embedding = layers.Dense(latent_dim // 2,
                                            activation=tf.nn.relu6)(z_mean)
    classification_embedding = layers.Dropout(0.2)(classification_embedding)
    classification_output = layers.Dense(num_class,
                                         activation="sigmoid",
                                         name="classification_output")(classification_embedding)

    model = Model(input_tensor, [recon_output,
                                 seg_output,
                                 classification_output])
    return model


def get_vae_model(input_shape, block_size=16,
                  encoder_output_filter=16,
                  groups=1, num_downsample=5,
                  base_act="relu", last_act="tanh",
                  latent_dim=16):
    padding = "same"

    ################################################
    ################# Define Layer #################
    ################################################
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    _, H, W, C = encoder.output.shape
    if encoder_output_filter is None:
        encoder_output_filter = C
    mean_std_layer = MeanSTD(latent_dim=latent_dim, name="mean_var")
    sampling_layer = Sampling(name="sampling")

    decode_dense_layer = layers.Dense(H * W * encoder_output_filter,
                                      activation=tf.nn.relu6)

    ################################################
    ################# Define call ##################
    ################################################
    input_tensor = encoder.input
    encoder_output = encoder(input_tensor)

    z_mean, z_log_var = mean_std_layer(encoder_output)
    z = sampling_layer([z_mean, z_log_var])
    z = decode_dense_layer(z)
    z = backend.reshape(z, (-1, H, W, encoder_output_filter))
    recon_output = HighWayDecoder2D(input_tensor=z,
                                    encoder=None, skip_connection_layer_names=None,
                                    last_filter=input_shape[-1],
                                    block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                    base_act=base_act, last_act=last_act, name_prefix="recon")

    model = Model(input_tensor, recon_output)
    return model


class MeanSTD(layers.Layer):
    def __init__(self, latent_dim, name=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = ""
        else:
            name = name + "_"
        self.flatten_layer = layers.Flatten()
        self.latent_dense_layer = layers.Dense(latent_dim * 4,
                                               activation=tf.nn.relu6)
        self.mean_dense_layer = layers.Dense(latent_dim, name=f"{name}z_mean")
        self.log_var_dense_layer = layers.Dense(
            latent_dim, name=f"{name}z_log_var")

    def call(self, x):
        flattend = self.flatten_layer(x)
        latent = self.latent_dense_layer(flattend)
        z_mean = self.mean_dense_layer(latent)
        z_log_var = self.log_var_dense_layer(latent)

        return z_mean, z_log_var


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_vq_vae_model(input_shape, block_size=16,
                     encoder_output_filter=16,
                     groups=1, num_downsample=5,
                     base_act="relu", last_act="tanh",
                     latent_dim=16, num_embeddings=64):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    vq_layer = VectorQuantizer(num_embeddings, embedding_dim=latent_dim,
                               name="vector_quantizer")
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    _, H, W, C = encoder.output.shape
    ################################################
    ################# Define call ##################
    ################################################
    input_tensor = encoder.input
    encoder_output = encoder(input_tensor)

    quantized_latents = vq_layer(encoder_output)

    recon_output = HighWayDecoder2D(input_tensor=quantized_latents,
                                    encoder=None, skip_connection_layer_names=None,
                                    last_filter=input_shape[-1],
                                    block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                    base_act=base_act, last_act=last_act, name_prefix="recon")

    model = Model(input_tensor, recon_output)
    return model


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
        _, H, W, C = backend.int_shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, [-1, H, W, C])

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
