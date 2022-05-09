import tensorflow as tf
from .base_model_resnet import HighWayResnet2D, HighWayDecoder2D
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend


def get_multi_scale_task_model(input_shape, num_class, block_size=16,
                               groups=10, num_downsample=5,
                               base_act="relu", last_act="tanh",
                               latent_dim=16, num_embeddings=64):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    vq_layer = VectorQuantizer(num_embeddings, embedding_dim=latent_dim,
                               name="vector_quantizer")
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=latent_dim,
                                                           groups=groups, num_downsample=num_downsample, padding=padding,
                                                           base_act=base_act, last_act=base_act)
    _, H, W, C = encoder.output.shape
    ################################################
    ################# Define call ##################
    ################################################
    input_tensor = encoder.input
    encoder_output = encoder(input_tensor)

    quantized_latents = vq_layer(encoder_output)

    classification_embedding = layers.GlobalAveragePooling2D()(encoder_output)
    classification_embedding = layers.Dense(C // 2)(classification_embedding)
    classification_embedding = layers.Dropout(0.2)(classification_embedding)
    classification_embedding = layers.Activation(
        tf.nn.relu6)(classification_embedding)

    classification_output = layers.Dense(
        num_class, activation="sigmoid")(classification_embedding)
    seg_output = HighWayDecoder2D(input_tensor=encoder_output, encoder=encoder,
                                  skip_connection_layer_names=SKIP_CONNECTION_LAYER_NAMES,
                                  last_filter=num_class,
                                  block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act="sigmoid", name_prefix="seg")
    recon_output = HighWayDecoder2D(input_tensor=quantized_latents,
                                    encoder=None, skip_connection_layer_names=None,
                                    last_filter=input_shape[-1],
                                    block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                    base_act=base_act, last_act=last_act, name_prefix="recon")

    model = Model(input_tensor, [classification_output,
                                 seg_output,
                                 recon_output])
    return model


def get_vq_vae_model(input_shape, block_size=16,
                     groups=1, num_downsample=5,
                     base_act="relu", last_act="tanh",
                     latent_dim=16, num_embeddings=64):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    vq_layer = VectorQuantizer(num_embeddings, embedding_dim=latent_dim,
                               name="vector_quantizer")
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D(input_shape=input_shape, block_size=block_size, last_filter=latent_dim,
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
