from .base_model import InceptionResNetV2, conv2d_bn, SKIP_CONNECTION_LAYER_NAMES
from .base_model_3d import InceptionResNetV2 as InceptionResNetV2_3D
from .base_model_as_class import InceptionResNetV2_progressive
from .layers import SkipUpsample3D, OutputLayer3D, TransformerEncoder, AddPositionEmbs, Decoder3D, Decoder2D, OutputLayer2D
from .layers import inception_resnet_block_3d, conv3d_bn, get_transformer_layer, HighwayMulti
from .layers_resnet import highway_conv2d, highway_decode2d
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import math


class ProgressiveX2CT():

    def __init__(self, final_xray_shape, final_ct_series_shape, init_filter,
                 block_size=16, start_downsample=2, final_downsample=5,
                 decode_init_filter=64,
                 base_act="leakyrelu", gen_last_act="tanh", disc_last_act="tanh"):
        ############################################
        ############## Define Attribute ############
        ############################################
        self.init_filter = init_filter
        self.block_size = block_size
        self.base_act = base_act
        self.start_downsample = start_downsample
        self.final_downsample = final_downsample
        self.decode_init_filter = decode_init_filter
        self.current_idx = final_downsample - start_downsample
        self.gen_last_act = gen_last_act
        self.disc_last_act = disc_last_act
        self.gen_name_prefix = "gen"
        self.disc_name_prefix = "disc"

        self.ct_dim = final_ct_series_shape[0] // (2 ** self.current_idx)
        self.xray_shape = np.array([self.ct_dim,
                                    self.ct_dim,
                                    final_xray_shape[-1]])
        self.ct_series_shape = np.array(
            final_ct_series_shape) // (2 ** self.current_idx)
        self.downsample_input_shape = np.array([self.ct_dim,
                                               self.ct_dim,
                                               (self.block_size * 2) * (2 ** self.current_idx)])
        self.upsample_input_shape = np.array([self.ct_dim,
                                             self.ct_dim,
                                             self.ct_dim,
                                             self.decode_init_filter * (2 ** self.current_idx)])
        ############################################
        ############ Define Base Model #############
        ############################################
        self.gen, self.disc = self.get_base_model()
        ############################################
        ############ Define Block List #############
        ############################################
        self.gen_downsample_block = []
        self.gen_upsample_block = []
        self.disc_downsample_block = []

    def get_model(self):
        return self.gen, self.disc

    def grow_model(self):
        self.grow_shape()
        self.grow_gen()
        self.grow_disc()

    # self.gen.layers: [input_layer, init_conv_block, 2d_3d_block]
    # 2d_3d_block.layers: [input_layer, 2d_3d_block, output_block]
    def grow_gen(self):
        filter_size = (self.block_size * 2) * (2 ** self.current_idx)
        gen_init_conv = self.get_init_conv(self.gen_name_prefix)
        downsample_block = self.build_downsample_block(self.gen_name_prefix)
        target_gen_model = Model(self.gen.layers[2].input,
                                 self.gen.layers[2].layers[1].output)
        upsample_block = self.build_upsample_block(self.gen_name_prefix,
                                                   downsample_block)
        output_block = self.build_output_block(self.gen_name_prefix,
                                               upsample_block)
        print(self.current_idx)
        print(downsample_block.input.shape)
        print(downsample_block.output.shape)
        print(target_gen_model.input.shape)
        print(target_gen_model.output.shape)
        input_tensor = gen_init_conv.input
        encoded = gen_init_conv(input_tensor)
        encoded = downsample_block(encoded)
        encoded_previous = target_gen_model(encoded)

        skip_connect = SkipUpsample3D(filter_size,
                                      activation=self.base_act)(encoded,
                                                                self.ct_dim // 2)
        concatenated = layers.Concatenate(axis=-1)([encoded_previous,
                                                    skip_connect])
        upsampled = upsample_block(concatenated)
        output = output_block(upsampled)
        self.gen = Model(input_tensor, output,
                         name=f"gen_{self.ct_dim}")

    # self.disc.layers: [input_layer, init_conv_block, encoder_block]
    # encoder_block.layers: [input_layer, init_conv_block, encoder_block]
    def grow_disc(self):
        disc_init_conv = self.get_init_conv(self.disc_name_prefix)
        downsample_block = self.build_downsample_block(self.disc_name_prefix)

        target_disc_model = Model(self.disc.layers[2].input,
                                  self.disc.layers[2].output)
        input_tensor = disc_init_conv.input
        encoded = disc_init_conv(input_tensor)
        encoded = downsample_block(encoded)
        encoded_previous = target_disc_model(encoded)

        self.disc = Model(input_tensor, encoded_previous,
                          name=f"disc_{self.ct_dim}")

    def grow_shape(self):
        self.ct_dim *= 2
        self.xray_shape = np.array([self.ct_dim,
                                    self.ct_dim,
                                    self.xray_shape[-1]])
        self.ct_series_shape *= 2
        self.current_idx += 1
        self.downsample_input_shape = np.array([self.ct_dim,
                                               self.ct_dim,
                                               (self.block_size * 2) * (2 ** self.current_idx)])
        self.upsample_input_shape = np.array([self.ct_dim,
                                             self.ct_dim,
                                             self.ct_dim,
                                             self.decode_init_filter * (2 ** self.current_idx)])

    def build_downsample_block(self, name_prefix):
        idx = self.current_idx
        layer_idx = self.final_downsample - idx - 1
        filter_size = (self.block_size * 2) * (2 ** idx)
        input_tensor = layers.Input(self.downsample_input_shape)
        x = highway_conv2d(input_tensor=input_tensor, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding="same", activation=self.base_act,
                           groups=1, name=f"{name_prefix}conv_down_{layer_idx}_0")
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding="same", activation=self.base_act, groups=1,
                           name=f"{name_prefix}conv_down_same_{layer_idx}_1")
        downsample_output = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                                           padding="same", activation=self.base_act,
                                           groups=1, name=f"{name_prefix}conv_down_{layer_idx}")
        return Model(input_tensor, downsample_output,
                     name=f"{name_prefix}_downsample_block_{layer_idx}")

    def build_upsample_block(self, name_prefix, downsample_block):
        idx = self.current_idx
        current_filter = self.decode_init_filter // (2 ** idx)
        input_shape = self.upsample_input_shape
        skip_connect_shape = downsample_block.output.shape[1:]
        input_shape[-1] = input_shape[-1] + skip_connect_shape[-1]
        input_tensor = layers.Input(input_shape)

        upsampled = conv3d_bn(input_tensor, current_filter, 3,
                              activation=self.base_act)
        upsampled = conv3d_bn(upsampled, current_filter, 3,
                              activation=self.base_act)
        upsampled = Decoder3D(current_filter,
                              strides=(2, 2, 2),
                              activation=self.base_act)(upsampled)
        return Model(input_tensor, upsampled,
                     name=f"{name_prefix}_upsample_block_{idx}")

    def build_output_block(self, name_prefix, upsample_block):
        idx = self.current_idx
        current_filter = self.decode_init_filter * (2 ** idx)
        output_block_input = layers.Input(upsample_block.output.shape[1:])
        output_tensor = conv3d_bn(output_block_input, current_filter, 3,
                                  activation=self.base_act)
        output_tensor = OutputLayer3D(last_channel_num=1,
                                      act=self.gen_last_act)(output_tensor)
        output_tensor = backend.squeeze(output_tensor, axis=-1)
        return Model(output_block_input, output_tensor,
                     name=f"{name_prefix}_output_block_{idx}")

    def get_init_conv(self, name_prefix):
        input_tensor = layers.Input(self.xray_shape)
        if self.current_idx <= self.final_downsample:
            init_filter = (self.block_size * 2) * (2 ** self.current_idx)
        else:
            init_filter = self.init_filter
        x = highway_conv2d(input_tensor=input_tensor, filters=init_filter,
                           downsample=False, same_channel=False,
                           padding="same", activation=self.base_act,
                           groups=1)
        return Model(input_tensor, x, name=f"{name_prefix}_init_conv")

    def get_base_model(self):
        gen_init_conv = self.get_init_conv(self.gen_name_prefix)
        disc_init_conv = self.get_init_conv(self.disc_name_prefix)

        gen_base = get_x2ct_model_progressive(gen_init_conv.output.shape[1:], self.ct_series_shape,
                                              block_size=self.block_size,
                                              num_downsample=self.start_downsample,
                                              final_downsample=self.final_downsample,
                                              decode_init_filter=self.decode_init_filter,
                                              base_act=self.base_act, last_act=self.gen_last_act,
                                              name_prefix=self.gen_name_prefix)

        gen_input_tensor = gen_init_conv.input
        gen_encoded = gen_init_conv(gen_input_tensor)
        gen_encoded = gen_base(gen_encoded)
        gen = Model(gen_input_tensor, gen_encoded,
                    name=f"gen_{self.ct_dim}")
        disc_base = get_resnet_disc_progressive(input_shape=disc_init_conv.output.shape[1:],
                                                num_downsample=self.start_downsample,
                                                final_downsample=self.final_downsample,
                                                block_size=self.block_size,
                                                base_act=self.base_act, last_act=self.disc_last_act,
                                                name_prefix=self.disc_name_prefix)
        disc_input_tensor = disc_init_conv.input
        disc_encoded = disc_init_conv(disc_input_tensor)
        disc_encoded = disc_base(disc_encoded)
        disc = Model(disc_input_tensor, disc_encoded,
                     name=f"disc_{self.ct_dim}")
        return gen, disc


def get_x2ct_model_progressive(xray_shape, ct_series_shape,
                               block_size=16,
                               num_downsample=5,
                               final_downsample=5,
                               decode_init_filter=64,
                               base_act="relu", last_act="tanh",
                               name_prefix=None):

    model, skip_connect_name_list = InceptionResNetV2_progressive(input_shape=(xray_shape[0], xray_shape[1], xray_shape[2]),
                                                                  block_size=block_size,
                                                                  num_downsample=num_downsample,
                                                                  final_downsample=final_downsample,
                                                                  padding="same",
                                                                  base_act=base_act,
                                                                  last_act=base_act,
                                                                  name_prefix=name_prefix)


InceptionResNetV2_progressive(target_shape=target_shape,
                              block_size=block_size,
                              padding="same",
                              base_act=base_act,
                              last_act=base_act,
                              name_prefix="x2ct",
                              num_downsample=num_downsample,
                              use_attention=True)
# ap_input.shape [B, 512, 512, 1]
# ap_output.shape: [B, 16, 16, 1536]
model_input = model.input
model_output = model.output
skip_connection_outputs = [model.get_layer(layer_name).output
                           for layer_name in skip_connect_name_list]

# lat_output.shape: [B, 16, 16, 1536]
_, H, W, C = backend.int_shape(model_output)

ct_start_channel = xray_shape[0] // (2 ** num_downsample)
ct_dim = ct_start_channel

attn_num_head = 8
attn_dim = C // attn_num_head

decoded = layers.Reshape((H * W, C))(model_output)
decoded = AddPositionEmbs(input_shape=(H * W, C))(decoded)

 attn_dim_list = [attn_dim for _ in range(6)]
  num_head_list = [attn_num_head for _ in range(6)]

   for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                     dropout=0)(decoded)

    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3)))(decoded)

    init_filter = decode_init_filter
    for idx in range(0, num_downsample, 1):
        current_idx = final_downsample - idx - 1
        current_filter = init_filter // (2 ** current_idx)

        skip_connect = skip_connection_outputs[-(idx + 1)]
        skip_connect = SkipUpsample3D(current_filter,
                                      activation=base_act)(skip_connect,
                                                           ct_dim)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3,
                            activation=base_act)
        decoded = conv3d_bn(decoded, current_filter, 3,
                            activation=base_act)

        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
        ct_dim *= 2

    encode_block = Model(model_input, decoded, name="2d_3d")

    output_block_input = layers.Input(encode_block.output.shape[1:])
    output_tensor = conv3d_bn(output_block_input, current_filter, 3,
                              activation=base_act)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)
    output_block = Model(output_block_input, output_tensor, name="output")

    input_tensor = encode_block.input
    convert_2d_3d = encode_block(input_tensor)
    convert_2d_3d = output_block(convert_2d_3d)

    return Model(input_tensor, convert_2d_3d)

def get_x2ct_model_progressive(xray_shape, ct_series_shape,
                               block_size=16,
                               num_downsample=5,
                               base_act="relu", last_act="tanh",
                               name_prefix=None):

def get_x2ct_model_progressive(xray_shape, ct_series_shape,
                              block_size=16,
                              num_downsample=5,
                              base_act="leakyrelu",
                              last_act="tanh",
                              name_prefix=None):
    filter_list = [block_size * 2, block_size * 4, block_size * 12,
                   block_size * 68, block_size * 130]
    model = InceptionResNetV2_progressive(target_shape=xray_shape,
                                          block_size=block_size,
                                          padding="same",
                                          base_act=base_act,
                                          last_act=base_act,
                                          name_prefix="x2ct",
                                          num_downsample=num_downsample,
                                          use_attention=True)
    # ap_input.shape [B, 512, 512, 1]
    # ap_output.shape: [B, 16, 16, 1536]
    model_input = model.input
    model_output = model.output
    # lat_output.shape: [B, 16, 16, 1536]
    _, H, W, C = backend.int_shape(model_output)
    ct_start_channel = H
    ct_dim = ct_start_channel
    attn_num_head = 8
    attn_dim = C // attn_num_head

    decoded = layers.Reshape((H * W, C))(model_output)
    decoded = AddPositionEmbs(input_shape=(H * W, C))(decoded)

    attn_dim_list = [attn_dim for _ in range(6)]
    num_head_list = [attn_num_head for _ in range(6)]

    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        decoded = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                     dropout=0)(decoded)
    decoded = layers.Reshape((ct_start_channel,
                              ct_start_channel,
                              ct_start_channel,
                              H * W * C // (ct_start_channel ** 3)))(decoded)
    decoded = conv3d_bn(decoded, C, 3, activation=base_act)
    for idx in range(5, 5 - num_downsample, -1):
        skip_connect = model.get_layer(f"x2ct_down_block_{idx}").output
        _, H, W, current_filter = skip_connect.shape
        skip_connect = SkipUpsample3D(current_filter,
                                      activation=base_act)(skip_connect, H)
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)
        decoded = layers.Concatenate()([decoded, skip_connect])
        decoded = conv3d_bn(decoded, current_filter, 3, activation=base_act)

        decoded = Decoder3D(current_filter,
                            strides=(2, 2, 2),
                            activation=base_act)(decoded)
    output_tensor = conv3d_bn(decoded, current_filter, 3, activation=base_act)
    output_tensor = conv3d_bn(output_tensor, current_filter // 2, 1, 
    activation=None)
    output_tensor = OutputLayer3D(last_channel_num=1,
                                  act=last_act)(output_tensor)
    output_tensor = backend.squeeze(output_tensor, axis=-1)

    return Model(model_input, output_tensor)


def get_resnet_disc_progressive(input_shape,
                                num_downsample=5,
                                final_downsample=5,
                                block_size=16,
                                base_act="leakyrelu", last_act="tanh",
                                name_prefix=None):
    padding = "same"
    encoder_output_filter = None
    groups = 1
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, SKIP_CONNECTION_LAYER_NAMES = HighWayResnet2D_Progressive(input_shape=input_shape, block_size=block_size, last_filter=encoder_output_filter,
                                                                       groups=groups, num_downsample=num_downsample, final_downsample=final_downsample,
                                                                       padding=padding, base_act=base_act, last_act=base_act,
                                                                       name_prefix=name_prefix)
    base_input = encoder.input

    # add a global spatial average pooling layer
    x = encoder.output
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(1, activation=last_act)(x)
    model = Model(base_input, predictions)
    return model
