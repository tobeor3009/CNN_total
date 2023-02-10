from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_as_class import InceptionResNetV2_progressive, Conv2DBN
from .base_model_as_class_3d import EqualizedConv3D, InceptionResNetV2_3D_Progressive, Conv3DBN
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from ..vision_transformer_cnn.swin_layers import SwinTransformerBlock3D

from .layers import SkipUpsample3D, SimpleOutputLayer3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import get_act_layer, conv3d_bn, get_transformer_layer, HighwayMulti, EqualizedDense
from .layers import PixelShuffleBlock3D, UpsampleBlock3D, TransposeBlock3D, SimpleOutputLayer2D, EqualizedConv, UpsampleBlock2D
from .multi_scale_task import MeanSTD, Sampling
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import math


def get_xray_encoder(xray_shape, block_size=32, latent_dim=16384,
                     num_downsample=3, base_act="gelu", base_norm="instance"):
    model_input = layers.Input(xray_shape)
    encoded = model_input
    for down_idx in range(num_downsample):
        block_scale = 2 ** down_idx
        current_block_size = block_size * block_scale
        encoded = Conv2DBN(current_block_size, 3, strides=2,
                           norm=base_norm, activation=base_act)(encoded)
    return Model(model_input, encoded)


def get_xray_decoder(xray_shape, block_size=32, latent_dim=16384,
                     num_upsample=3, base_act="gelu", base_norm="instance", last_act="sigmoid"):
    model_input = layers.Input((latent_dim,))
    init_decode_shape = (xray_shape[0] // (2 ** num_upsample),
                         xray_shape[1] // (2 ** num_upsample),
                         xray_shape[2] // (2 ** num_upsample),
                         -1)
    decoded = layers.Reshape(init_decode_shape)(model_input)
    decoded = Conv2DBN(block_size, 3,
                       norm=base_norm, activation=base_act)(decoded)
    for up_idx in range(num_upsample):
        block_scale = 2 ** up_idx
        current_block_size = block_size * block_scale
        decoded = UpsampleBlock2D(current_block_size, strides=(2, 2),
                                  norm=base_norm, activation=base_act)(decoded)
    decoded = SimpleOutputLayer2D(last_channel_num=1,
                                  act=last_act)(decoded)
    return Model(model_input, decoded)


def get_ct_encoder(ct_shape, block_size=32, latent_dim=16384,
                   num_downsample=3, base_act="gelu", base_norm="instance"):
    Z, H, W, C = ct_shape
    mean_std_layer = MeanSTD(latent_dim=latent_dim, name=None)
    sampling_layer = Sampling(name=None)
    decode_dense_layer = layers.Dense(Z * H * W * block_size // (2 ** (3 * num_downsample)),
                                      activation=tf.nn.relu6)
    reshape_layer = layers.Reshape((Z // (2 ** num_downsample),
                                    H // (2 ** num_downsample),
                                    W // (2 ** num_downsample),
                                    block_size))
    model_input = layers.Input(ct_shape)
    encoded = model_input
    for down_idx in range(num_downsample):
        block_scale = 2 ** down_idx
        current_block_size = block_size * block_scale
        encoded = Conv3DBN(current_block_size, 3, strides=2,
                           norm=base_norm, activation=base_act)(encoded)
        encoded = SwinTransformerBlock3D(current_block_size,
                                         num_patch=[Z // (2 ** (down_idx + 1)),
                                                    H // (2 ** (down_idx + 1)),
                                                    W // (2 ** (down_idx + 1))],
                                         window_size=[2, 2, 2], num_heads=8,
                                         swin_v2=True, num_mlp=512)(encoded)
    encoded = Conv3DBN(block_size, 3, strides=1,
                       norm=base_norm, activation=base_act)(encoded)
    z_mean, z_log_var = mean_std_layer(encoded)
    sampling_z = sampling_layer([z_mean, z_log_var])
    encoded = decode_dense_layer(sampling_z)
    encoded = reshape_layer(encoded)
    return Model(model_input, encoded)


def get_ct_disc(ct_shape, block_size=32, latent_dim=16384,
                num_downsample=3, base_act="gelu", base_norm="instance"):
    model_input = layers.Input(ct_shape)
    encoded = model_input
    for down_idx in range(num_downsample):
        block_scale = 2 ** down_idx
        current_block_size = block_size * block_scale
        encoded = Conv3DBN(current_block_size, 3, strides=2,
                           norm=base_norm, activation=base_act)(encoded)
    return Model(model_input, encoded)


def get_ct_decoder(ct_shape, block_size=32, latent_dim=16384,
                   num_upsample=3, base_act="gelu", base_norm="instance", last_act="sigmoid"):
    init_decode_shape = (ct_shape[0] // (2 ** num_upsample),
                         ct_shape[1] // (2 ** num_upsample),
                         ct_shape[2] // (2 ** num_upsample),
                         block_size)
    Z, H, W, C = init_decode_shape
    model_input = layers.Input(init_decode_shape)
    decoded = model_input
    for up_idx in range(num_upsample):
        block_scale = num_upsample - up_idx
        current_block_size = block_size * block_scale
        upsample = UpsampleBlock3D(current_block_size, strides=(2, 2, 2),
                                   norm=base_norm, activation=base_act)(decoded)
        pixel_shuffle = PixelShuffleBlock3D(current_block_size, strides=(2, 2, 2),
                                            norm=base_norm, activation=base_act)(decoded)
        decoded = layers.Concatenate(axis=-1)([upsample, pixel_shuffle])
        decoded = Conv3DBN(current_block_size, 3, strides=1,
                           norm=base_norm, activation=base_act)(decoded)
        # decoded = WindowAttention3D(current_block_size, window_size=[2, 2, 2], num_heads=8,
        #                             swin_v2=True)(decoded, decoded)
    decoded = SimpleOutputLayer3D(last_channel_num=1,
                                  act=last_act)(decoded)
    return Model(model_input, decoded)


class SelfSpatialAttention3D(layers.Layer):
    def __init__(self, Z, H, W, channel):
        super().__init__()
        self.input_bn = layers.BatchNormalization()
        self.output_bn = layers.BatchNormalization()
        self.conv_q = layers.Conv3D(channel // 8,
                                    kernel_size=1, padding="same")
        self.conv_k = layers.Conv3D(channel // 8,
                                    kernel_size=1, padding="same")
        self.conv_v = layers.Conv3D(channel // 8,
                                    kernel_size=1, padding="same")
        self.conv_proj = layers.Conv3D(channel, kernel_size=1, padding="same")
        self.reshape_3d_1d = layers.Reshape((Z * H * W, channel // 8))
        self.permute_layer = layers.Permute(dims=[2, 1])
        self.softmax_layer = get_act_layer("softmax")
        self.dot_layer_1 = layers.Dot(axes=[1, 2])
        self.dot_layer_2 = layers.Dot(axes=[1, 2])
        self.reshape_1d_3d = layers.Reshape((Z, H, W, channel // 8))

    def call(self, input_tensor):
        img = self.input_bn(input_tensor)
        q = self.conv_q(img)
        k = self.conv_k(img)
        v = self.conv_v(img)
        q = self.reshape_3d_1d(q)
        k = self.reshape_3d_1d(k)
        k = self.permute_layer(k)
        v = self.reshape_3d_1d(v)

        # should we scale this?
        s = self.dot_layer_1([q, k])
        beta = self.softmax_layer(s)
        attn = self.dot_layer_2([beta, v])
        out = self.reshape_1d_3d(attn)
        out = self.conv_proj(out)
        out = self.output_bn(out)
        return out


class XrayReconsturct(Model):
    def __init__(self, xray_shape, block_size=32, latent_dim=16384, num_updownsample=3,
                 base_act="gelu", base_norm="instance", last_act="sigmoid"):
        super().__init__()
        self.encoder = get_xray_encoder(xray_shape, block_size, latent_dim,
                                        num_updownsample, base_act, base_norm)
        self.decoder = get_xray_decoder(xray_shape, block_size, latent_dim,
                                        num_updownsample, base_act, base_norm, last_act)

    def call(self, inputs, training=False, return_latent=False):
        encoded_latent = self.encoder(inputs)
        decoded = self.decoder(encoded_latent)
        if return_latent:
            return encoded_latent, decoded
        else:
            return decoded


class XrayDiscriminator(Model):
    def __init__(self, xray_shape, block_size=32, latent_dim=32, num_updownsample=3,
                 base_act="gelu", base_norm="instance", last_act="sigmoid"):
        super().__init__()
        self.encoder = get_xray_encoder(xray_shape, block_size, latent_dim,
                                        num_updownsample, base_act, base_norm)
        self.act = get_act_layer(last_act)

    def call(self, inputs, training=False):
        encoded_latent = self.encoder(inputs)
        encoded_latent = self.act(encoded_latent)
        return encoded_latent


class CTReconsturct(Model):
    def __init__(self, ct_shape, block_size=32, latent_dim=16384, num_updownsample=3,
                 base_act="gelu", base_norm="instance", last_act="sigmoid"):
        super().__init__()
        self.encoder = get_ct_encoder(ct_shape, block_size, latent_dim,
                                      num_updownsample, base_act, base_norm)
        self.decoder = get_ct_decoder(ct_shape, block_size, latent_dim,
                                      num_updownsample, base_act, base_norm, last_act)

    def call(self, inputs, training=False, return_latent=False):
        encoded_latent = self.encoder(inputs)
        decoded = self.decoder(encoded_latent)
        if return_latent:
            return encoded_latent, decoded
        else:
            return decoded


class CTDiscriminator(Model):
    def __init__(self, ct_shape, block_size=32, latent_dim=32, num_updownsample=3,
                 base_act="gelu", base_norm="instance", last_act="sigmoid"):
        super().__init__()
        self.encoder = get_ct_disc(ct_shape, block_size, latent_dim,
                                   num_updownsample, base_act, base_norm)
        self.conv = layers.Conv3D(1, kernel_size=1, padding="same")
        self.act = get_act_layer(last_act)

    def call(self, inputs, training=False):
        encoded_latent = self.encoder(inputs)
        encoded_latent = self.conv(encoded_latent)
        encoded_latent = self.act(encoded_latent)
        return encoded_latent


class Xray2CT(Model):
    def __init__(self, xray_shape, ct_shape, block_size=32, latent_dim=32, num_updownsample=3,
                 base_act="gelu", base_norm="instance", last_act="sigmoid"):
        super().__init__()
        self.encoder_2d = get_xray_encoder(xray_shape, block_size, latent_dim,
                                           num_updownsample, base_act, base_norm)
        self.decode_init_h = xray_shape[0] // (2 ** num_updownsample)
        self.decode_init_w = xray_shape[1] // (2 ** num_updownsample)
        self.decode_init_z = self.decode_init_h // 4
        self.decode_channel = block_size * (2 ** (num_updownsample - 1))
        self.decode_init_channel = self.decode_channel // self.decode_init_z
        self.reshape_2d_3d = layers.Reshape((self.decode_init_z,
                                             self.decode_init_h, self.decode_init_w,
                                             self.decode_channel // (self.decode_init_z)))
        self.middle_conv_1 = Conv3DBN(self.decode_init_channel * 2, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_1 = SelfSpatialAttention3D(self.decode_init_z,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 2)
        self.upsample_3d_1 = layers.UpSampling3D(size=(2, 1, 1))
        self.middle_conv_2 = Conv3DBN(self.decode_init_channel * 4, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_2 = SelfSpatialAttention3D(self.decode_init_z * 2,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 4)
        self.upsample_3d_2 = layers.UpSampling3D(size=(2, 1, 1))

        self.middle_conv_3 = Conv3DBN(self.decode_channel, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_3 = SelfSpatialAttention3D(self.decode_init_z * 4,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_channel)
        self.decoder_3d = get_ct_decoder(ct_shape, block_size, latent_dim,
                                         num_updownsample, base_act, base_norm, last_act)

    def call(self, inputs):
        encoded_2d = self.encoder_2d(inputs)
        decoded_3d = self.reshape_2d_3d(encoded_2d)
        decoded_3d = self.middle_conv_1(decoded_3d)
        decoded_3d = self.middle_attn_1(decoded_3d)
        decoded_3d = self.upsample_3d_1(decoded_3d)
        decoded_3d = self.middle_conv_2(decoded_3d)
        decoded_3d = self.middle_attn_2(decoded_3d)
        decoded_3d = self.upsample_3d_2(decoded_3d)
        decoded_3d = self.middle_conv_3(decoded_3d)
        decoded_3d = self.middle_attn_3(decoded_3d)
        decoded_3d = self.decoder_3d(decoded_3d)
        return decoded_3d


class Xray2CTLatent(Model):
    def __init__(self, xray_shape, block_size=32, latent_dim=32, num_updownsample=3,
                 base_act="gelu", base_norm="instance"):
        super().__init__()

        self.encoder_2d = get_xray_encoder(xray_shape, block_size, latent_dim,
                                           num_updownsample, base_act, base_norm)
        self.decode_init_h = xray_shape[0] // (2 ** num_updownsample)
        self.decode_init_w = xray_shape[1] // (2 ** num_updownsample)
        self.decode_init_z = self.decode_init_h // 4
        self.decode_channel = block_size * (2 ** (num_updownsample - 1))
        self.decode_init_channel = self.decode_channel // self.decode_init_z
        self.reshape_2d_3d = layers.Reshape((self.decode_init_z,
                                             self.decode_init_h, self.decode_init_w,
                                             self.decode_channel // (self.decode_init_z)))
        self.middle_conv_1 = Conv3DBN(self.decode_init_channel * 2, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_1 = SelfSpatialAttention3D(self.decode_init_z,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 2)
        self.upsample_3d_1 = layers.UpSampling3D(size=(2, 1, 1))
        self.middle_conv_2 = Conv3DBN(self.decode_init_channel * 4, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_2 = SelfSpatialAttention3D(self.decode_init_z * 2,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 4)
        self.upsample_3d_2 = layers.UpSampling3D(size=(2, 1, 1))

        self.middle_conv_3 = Conv3DBN(self.decode_channel, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_3 = SelfSpatialAttention3D(self.decode_init_z * 4,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_channel)

    def call(self, inputs):
        encoded_2d = self.encoder_2d(inputs)
        decoded_3d = self.reshape_2d_3d(encoded_2d)
        decoded_3d = self.middle_conv_1(decoded_3d)
        decoded_3d = self.middle_attn_1(decoded_3d)
        decoded_3d = self.upsample_3d_1(decoded_3d)
        decoded_3d = self.middle_conv_2(decoded_3d)
        decoded_3d = self.middle_attn_2(decoded_3d)
        decoded_3d = self.upsample_3d_2(decoded_3d)
        decoded_3d = self.middle_conv_3(decoded_3d)
        decoded_3d = self.middle_attn_3(decoded_3d)
        return decoded_3d


class Xray2CTLatent(Model):
    def __init__(self, xray_shape, block_size=32, latent_dim=32, num_updownsample=3,
                 base_act="gelu", base_norm="instance"):
        super().__init__()
        self.encoder_2d = get_xray_encoder(xray_shape, block_size, latent_dim,
                                           num_updownsample, base_act, base_norm)
        self.decode_init_h = xray_shape[0] // (2 ** num_updownsample)
        self.decode_init_w = xray_shape[1] // (2 ** num_updownsample)
        self.decode_init_z = self.decode_init_h // 4
        self.decode_channel = block_size * (2 ** (num_updownsample - 1))
        self.decode_init_channel = self.decode_channel // self.decode_init_z
        self.encode_feature_shrink_conv = Conv2DBN(self.decode_channel // 4, 3, strides=1,
                                                   norm=base_norm, activation=base_act)
        self.reshape_2d_3d = SkipUpsample3D(self.decode_channel,
                                            norm=base_norm, activation=base_act)
        self.middle_conv_1 = Conv3DBN(self.decode_init_channel * 2, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_1 = SelfSpatialAttention3D(self.decode_init_z,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 2)
        self.upsample_3d_1 = layers.UpSampling3D(size=(2, 1, 1))
        self.middle_conv_2 = Conv3DBN(self.decode_init_channel * 4, 3, strides=1,
                                      norm=base_norm, activation=base_act)
        self.middle_attn_2 = SelfSpatialAttention3D(self.decode_init_z * 2,
                                                    self.decode_init_h,
                                                    self.decode_init_w,
                                                    self.decode_init_channel * 4)
        self.upsample_3d_2 = layers.UpSampling3D(size=(2, 1, 1))

        self.middle_conv_3 = Conv3DBN(self.decode_channel, 3, strides=1,
                                      norm=base_norm, activation=base_act)

    def call(self, inputs):
        encoded_2d = self.encoder_2d(inputs)
        decoded_3d = self.reshape_2d_3d(encoded_2d, self.decode_init_z)
        decoded_3d = self.middle_conv_1(decoded_3d)
        decoded_3d = self.middle_attn_1(decoded_3d)
        decoded_3d = self.upsample_3d_1(decoded_3d)
        decoded_3d = self.middle_conv_2(decoded_3d)
        decoded_3d = self.middle_attn_2(decoded_3d)
        decoded_3d = self.upsample_3d_2(decoded_3d)
        decoded_3d = self.middle_conv_3(decoded_3d)
        return decoded_3d


def skip_recon_vae_block(input_shape, latent_dim,
                         base_act, downscale=False, name=None):
    H, W, C = backend.int_shape(input_shape)

    mean_std_layer = MeanSTD(latent_dim=latent_dim, name=None)
    sampling_layer = Sampling(name=None)
    decode_dense_layer = layers.Dense(H * W * H * C,
                                      activation=tf.nn.relu6)
    reshape_layer = layers.Reshape((H, W, H,
                                    C // 4))
    model_input = layers.Input(input_shape[1:])
    if downscale:
        scale = Conv2DBN(model_input, C // 4, 3,
                         activation=base_act)
    else:
        scale = model_input
    z_mean, z_log_var = mean_std_layer(scale)
    sampling_z = sampling_layer([z_mean, z_log_var])
    decoded = decode_dense_layer(sampling_z)
    decoded = reshape_layer(decoded)
    decoded = Conv3DBN(decoded, C, 3, activation=base_act)
    return Model(model_input, decoded, name=name)
