from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_as_class import InceptionResNetV2_progressive, Conv2DBN
from .base_model_as_class_3d import EqualizedConv3D, InceptionResNetV2_3D_Progressive, Conv3DBN
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .layers import SkipUpsample3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import get_act_layer, conv3d_bn, get_transformer_layer, HighwayMulti, EqualizedDense, PixelShuffleBlock2D, SimpleOutputLayer3D
from .layers import PixelShuffleBlock3D, UpsampleBlock3D, TransposeBlock3D, SimpleOutputLayer2D, EqualizedConv, UpsampleBlock2D
from .multi_scale_task import MeanSTD, Sampling
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


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


def get_2d_encoder(input_shape, filter_size=32, dense_dim=16, latent_dim=64,
                   norm="instance", act="gelu"):

    encoder_inputs = layers.Input(shape=input_shape)
    x = Conv2DBN(filter_size, 3, strides=2, norm=norm,
                 activation=act)(encoder_inputs)
    x = Conv2DBN(filter_size * 2, 3, strides=2, norm=norm,
                 activation=act)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_dim, activation=tf.nn.relu6)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def get_2d_decoder(output_shape, latent_dim, dense_dim=64, filter_size=32,
                   norm="instance", act="gelu", last_act="sigmoid"):
    H, W, last_channel_num = output_shape
    h, w = H // 4, W // 4
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h * w * dense_dim, activation=tf.nn.relu6)(latent_inputs)
    x = layers.Reshape((h, w, dense_dim))(x)
    for decoder_idx in range(2, 0, -1):
        current_filter = filter_size * (2 ** decoder_idx)
        upsample = UpsampleBlock2D(current_filter,
                                   kernel_size=(2, 2),
                                   norm=norm, activation=act)(x)

        pixel_shuffle = PixelShuffleBlock2D(current_filter,
                                            kernel_size=2,
                                            norm=norm, activation=act)(x)
        x = layers.Concatenate()([upsample, pixel_shuffle])

    decoder_outputs = SimpleOutputLayer2D(last_channel_num=last_channel_num,
                                          act=last_act)(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def get_3d_encoder(input_shape, filter_size=32, dense_dim=16, latent_dim=64,
                   norm="instance", act="gelu"):

    encoder_inputs = layers.Input(shape=input_shape)
    x = Conv3DBN(filter_size, 3, strides=2, norm=norm,
                 activation=act)(encoder_inputs)
    x = Conv3DBN(filter_size * 2, 3, strides=2, norm=norm,
                 activation=act)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_dim, activation=tf.nn.relu6)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def get_3d_decoder(output_shape, latent_dim, dense_dim=64, filter_size=32,
                   norm="instance", act="gelu", last_act="sigmoid"):
    H, W, Z, last_channel_num = output_shape
    h, w, z = H // 4, W // 4, Z // 4
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h * w * dense_dim, activation=tf.nn.relu6)(latent_inputs)
    x = layers.Reshape((h, w, dense_dim))(x)
    for decoder_idx in range(2, 0, -1):
        current_filter = filter_size * (2 ** decoder_idx)
        upsample = UpsampleBlock3D(current_filter,
                                   kernel_size=(2, 2, 2),
                                   norm=norm, activation=act)(x)

        pixel_shuffle = PixelShuffleBlock3D(current_filter,
                                            kernel_size=(2, 2, 2),
                                            norm=norm, activation=act)(x)
        x = layers.Concatenate()([upsample, pixel_shuffle])

    decoder_outputs = SimpleOutputLayer3D(last_channel_num=last_channel_num,
                                          act=last_act)(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(Model):
    def __init__(self, encoder, decoder, image_loss_fn, train_2d, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.image_loss_fn = image_loss_fn
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.data_idx = 0 if train_2d else 1
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        source = data[self.data_idx]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(source)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.image_loss_fn(source, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss, axis=1)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
