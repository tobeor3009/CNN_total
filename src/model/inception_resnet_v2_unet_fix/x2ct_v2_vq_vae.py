from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_as_class import InceptionResNetV2_progressive, Conv2DBN
from .base_model_as_class_3d import EqualizedConv3D, InceptionResNetV2_3D_Progressive, Conv3DBN
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .layers import SkipUpsample3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import get_act_layer, conv3d_bn, get_transformer_layer, HighwayMulti, EqualizedDense, PixelShuffleBlock2D, SimpleOutputLayer3D
from .layers import PixelShuffleBlock3D, UpsampleBlock3D, TransposeBlock3D, SimpleOutputLayer2D, EqualizedConv, UpsampleBlock2D
from .multi_scale_task import MeanSTD, Sampling
from ..vision_transformer.classfication import swin_classification_2d_base, swin_classification_3d_base
from ..vision_transformer import swin_layers, transformer_layers, utils
from ..vision_transformer.base_layer import swin_transformer_stack_2d, swin_transformer_stack_3d
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import math

BLOCK_MODE_NAME = "seg"


def get_2d_3d_tensor(decoded, channel, norm, act, H, C, block_num=8):

    attn_dropout_proba = 0.1
    attn_dim_list = [channel // 8 for _ in range(block_num)]
    num_head_list = [8 for _ in range(block_num)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)

    decoded = Conv2DBN(channel, 3, norm=norm,
                       activation=act)(decoded)
    decoded = layers.Reshape((H ** 2, channel))(decoded)

    decoded = attn_sequence(decoded)
    decoded = layers.Reshape((H, H, H,
                              channel // H))(decoded)
    decoded = Conv3DBN(C, 3,
                       norm=norm, activation=act)(decoded)
    return decoded


class Block2D3D(layers.Layer):
    def __init__(self, channel, norm, act, block_num=6):
        super().__init__()
        self.channel = channel
        self.norm = norm
        self.act = act
        attn_dropout_proba = 0.1
        attn_dim_list = [channel // 8 for _ in range(block_num)]
        num_head_list = [8 for _ in range(block_num)]
        attn_layer_list = []
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=attn_dropout_proba)
            attn_layer_list.append(attn_layer)
        self.attn_sequence = Sequential(attn_layer_list)
        self.conv_2d = Conv2DBN(channel, 3, norm=norm, activation=act)

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape
        self.reshape_layer_1d = layers.Reshape((self.H * self.W,
                                                self.channel))
        self.reshape_layer_2d = layers.Reshape((self.H, self.H, self.H,
                                                self.channel // self.H))
        self.conv_3d = Conv3DBN(self.C, 3, norm=self.norm, activation=self.act)

    def call(self, input_tensor):
        decoded = self.conv_2d(input_tensor)
        decoded = self.reshape_layer_1d(decoded)
        decoded = self.attn_sequence(decoded)
        decoded = self.reshape_layer_2d(decoded)
        decoded = self.conv_3d(decoded)
        return decoded


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

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

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

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


def get_2d_encoder(input_shape,
                   filter_num_begin, depth, stack_num_down,
                   patch_size, stride_mode, num_heads, window_size, num_mlp,
                   act="gelu", shift_window=True, swin_v2=False):
    name = "encoder"
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    encoder_inputs = layers.Input(shape=input_shape)
    input_size = encoder_inputs.shape.as_list()[1:]
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_size[0:2],
                                                            patch_size,
                                                            stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    X = encoder_inputs
    # Patch extraction
    X = transformer_layers.PatchExtract(patch_size,
                                        stride_size)(X)
    print(f"PatchExtract shape: {X.shape}")
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim)(X)
    print(f"PatchEmbedding shape: {X.shape}")
    # The first Swin Transformer stack
    X = swin_transformer_stack_2d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(num_patch_x, num_patch_y),
                                  num_heads=num_heads[0],
                                  window_size=window_size[0],
                                  num_mlp=num_mlp,
                                  act=act,
                                  mode=BLOCK_MODE_NAME,
                                  shift_window=shift_window,
                                  swin_v2=swin_v2,
                                  name='{}_swin_down'.format(name))
    # Downsampling blocks
    for i in range(depth - 1):
        print(f"depth {i} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            name='down{}'.format(i))(X)
        print(f"depth {i} X merging shape: {X.shape}")

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[i + 1],
                                      window_size=window_size[i + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      name='{}_swin_down{}'.format(name, i + 1))

        print(f"depth {i} X Skip shape: {X.shape}")
    encoder = Model(encoder_inputs, X, name="encoder")
    return encoder


def get_2d_decoder(input_shape, last_channel_num, depth, stack_num_up, embed_dim,
                   patch_size, num_patch_x, num_patch_y, num_heads, window_size, num_mlp,
                   act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, name='unet'):
    decoder_inputs = layers.Input(shape=input_shape)
    X = decoder_inputs
    for i in range(depth):
        print(f"depth decode {i} X shape: {X.shape}")
        # Patch expanding
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              swin_v2=swin_v2,
                                              return_vector=True)(X)
        print(f"depth expanding {i} X shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        X = layers.Dense(embed_dim, use_bias=False,
                         name='{}_concat_linear_proj_{}'.format(name, i))(X)

        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[i],
                                      window_size=window_size[i],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      name='{}_swin_up{}'.format(name, i))
    # X = layers.Reshape((num_patch_x, num_patch_y, embed_dim))(X)
    X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                          embed_dim=embed_dim,
                                          upsample_rate=patch_size[0],
                                          swin_v2=swin_v2,
                                          return_vector=False)(X)
    X = layers.Conv2D(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act)(X)
    decoder = Model(decoder_inputs, X, name="decoder")
    return decoder


def get_3d_encoder(input_shape,
                   filter_num_begin, depth, stack_num_down,
                   patch_size, stride_mode, num_heads, window_size, num_mlp,
                   act="gelu", shift_window=True, swin_v2=False):
    name = "encoder"
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    encoder_inputs = layers.Input(shape=input_shape)
    input_size = encoder_inputs.shape.as_list()[1:]
    num_patch_z, num_patch_x, num_patch_y = utils.get_image_patch_num_3d(input_size[0:3],
                                                                         patch_size,
                                                                         stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    X = encoder_inputs
    # Patch extraction
    X = transformer_layers.PatchExtract3D(patch_size,
                                          stride_size)(X)
    print(f"PatchExtract shape: {X.shape}")
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_z * num_patch_x * num_patch_y,
                                          embed_dim)(X)
    print(f"PatchEmbedding shape: {X.shape}")
    # The first Swin Transformer stack
    X = swin_transformer_stack_3d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(num_patch_z,
                                             num_patch_x,
                                             num_patch_y),
                                  num_heads=num_heads[0],
                                  window_size=window_size[0],
                                  num_mlp=num_mlp,
                                  act=act,
                                  mode=BLOCK_MODE_NAME,
                                  shift_window=shift_window,
                                  swin_v2=swin_v2,
                                  name='{}_swin_down'.format(name))
    # Downsampling blocks
    for i in range(depth - 1):
        print(f"depth {i} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.PatchMerging3D((num_patch_z, num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              swin_v2=swin_v2,
                                              include_3d=True,
                                              name='down{}'.format(i))(X)
        print(f"depth {i} X merging shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_z = num_patch_z // 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_z,
                                                 num_patch_x,
                                                 num_patch_y),
                                      num_heads=num_heads[i + 1],
                                      window_size=window_size[i + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      name='{}_swin_down{}'.format(name, i + 1))

        print(f"depth {i} X Skip shape: {X.shape}")
    encoder = Model(encoder_inputs, X, name="encoder")
    return encoder


def get_3d_decoder(input_shape, last_channel_num, depth, stack_num_up, embed_dim,
                   patch_size, num_patch_z, num_patch_x, num_patch_y, num_heads, window_size, num_mlp,
                   act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, name='unet'):
    decoder_inputs = layers.Input(shape=input_shape)
    X = decoder_inputs
    for i in range(depth):
        print(f"depth decode {i} X shape: {X.shape}")
        # Patch expanding
        X = transformer_layers.PatchExpanding3D(num_patch=(num_patch_z,
                                                           num_patch_x,
                                                           num_patch_y),
                                                embed_dim=embed_dim,
                                                upsample_rate=2,
                                                swin_v2=swin_v2,
                                                return_vector=True)(X)
        print(f"depth expanding {i} X shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_z = num_patch_z * 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        X = layers.Dense(embed_dim, use_bias=False,
                         name='{}_concat_linear_proj_{}'.format(name, i))(X)

        # Swin Transformer stacks
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_z,
                                                 num_patch_x,
                                                 num_patch_y),
                                      num_heads=num_heads[i],
                                      window_size=window_size[i],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      name='{}_swin_up{}'.format(name, i))
        print(f"depth decode output {i} X shape: {X.shape}")
    # X = layers.Reshape((num_patch_x, num_patch_y, embed_dim))(X)
    X = transformer_layers.PatchExpanding3D(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            upsample_rate=patch_size[0],
                                            swin_v2=swin_v2,
                                            return_vector=False)(X)
    X = layers.Conv3D(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act)(X)
    decoder = Model(decoder_inputs, X, name="decoder")
    return decoder


def get_vqvae(encoder, decoder, num_embeddings):
    latent_dim = encoder.output.shape[-1]
    vq_layer = VectorQuantizer(num_embeddings, latent_dim,
                               name="vector_quantizer")
    input_tensor = layers.Input(encoder.input.shape[1:])
    encoder_outputs = encoder(input_tensor)
    print(f"encoded_shape: {encoder_outputs.shape}")
    quantized_latents = vq_layer(encoder_outputs)
    print(f"quantized_shape: {encoder_outputs.shape}")
    reconstructions = decoder(quantized_latents)
    return Model(input_tensor, reconstructions, name="vq_vae")


class VQVAETrainer(Model):
    def __init__(self, train_variance, encoder, decoder, num_embeddings, train_2d, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.output.shape[-1]
        self.num_embeddings = num_embeddings
        self.data_idx = 0 if train_2d else 1
        self.vqvae = get_vqvae(encoder, decoder, self.num_embeddings)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def call(self, x):
        return self.vqvae(x)

    def train_step(self, data):
        x = data[self.data_idx]
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) /
                self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
