import tensorflow as tf
from .base_model_as_class import InceptionResNetV2, InceptionResNetV2_SkipConnectLevel_0, InceptionResNetV2_SkipConnectLevel_1
from .base_model_as_class import Conv2DBN, DecoderBlock2D, DecoderBlock2D_MultiScale
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from itertools import zip_longest
from .util.pathology import recon_overlapping_patches


def get_multi_scale_task_model(input_shape, num_class, block_size=16,
                               groups=10, num_downsample=5,
                               base_act="relu", last_act="tanh",
                               latent_dim=16):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, skip_connect_layer_names = InceptionResNetV2(input_shape=input_shape, block_size=block_size, padding=padding,
                                                          groups=groups, base_act=base_act, last_act=base_act, name_prefix="",
                                                          skip_connect_names=True)
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
    x = Conv2DBN(C // 2, strides=2, padding=padding,
                 activation=base_act, groups=groups)(encoder_output)
    x = Conv2DBN(C // 4, strides=2, padding=padding,
                 activation=base_act, groups=groups)(x)
    z_mean, z_log_var = mean_std_layer(x)
    sampling_z = sampling_layer([z_mean, z_log_var])
    z = decode_dense_layer(sampling_z)
    z = backend.reshape(z, (-1, H // 4, W // 4, C // 4))
    recon_output = DecoderBlock2D(input_tensor=z, encoder=encoder,
                                  skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=False,
                                  last_channel_num=input_shape[-1],
                                  block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act=last_act, name_prefix="recon")
    seg_output = DecoderBlock2D(input_tensor=encoder_output, encoder=encoder,
                                skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
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


def get_multi_scale_task_model_v2(input_shape, num_class, block_size=16,
                                  groups=10, num_downsample=5,
                                  base_act="relu", last_act="tanh",
                                  latent_dim=16):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    encoder, skip_connect_layer_names = InceptionResNetV2(input_shape=input_shape, block_size=block_size, padding=padding,
                                                          groups=groups, base_act=base_act, last_act=base_act, name_prefix="",
                                                          skip_connect_names=True)
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
    x = Conv2DBN(C // 2, strides=2, padding=padding,
                 activation=base_act, groups=groups)(encoder_output)
    x = Conv2DBN(C // 4, strides=2, padding=padding,
                 activation=base_act, groups=groups)(x)
    z_mean, z_log_var = mean_std_layer(x)
    sampling_z = sampling_layer([z_mean, z_log_var])
    z = decode_dense_layer(sampling_z)
    z = backend.reshape(z, (-1, H // 4, W // 4, C // 4))
    recon_output = DecoderBlock2D(input_tensor=z, encoder=encoder,
                                  skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=False,
                                  last_channel_num=input_shape[-1],
                                  block_size=block_size, groups=1, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act=last_act, name_prefix="recon")
    seg_output = DecoderBlock2D(input_tensor=encoder_output, encoder=encoder,
                                skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=True,
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
                  groups=1, num_downsample=5,
                  base_act="relu", last_act="tanh",
                  latent_dim=16):
    padding = "same"

    ################################################
    ################# Define Layer #################
    ################################################
    encoder, skip_connect_layer_names = InceptionResNetV2(input_shape=input_shape, block_size=block_size, padding=padding,
                                                          groups=groups, base_act=base_act, last_act=base_act, name_prefix="",
                                                          skip_connect_names=True)
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
    x = Conv2DBN(C // 2, 3, strides=2, padding=padding,
                 activation=base_act, groups=groups)(encoder_output)
    x = Conv2DBN(C // 4, 3, strides=2, padding=padding,
                 activation=base_act, groups=groups)(x)
    z_mean, z_log_var = mean_std_layer(x)
    sampling_z = sampling_layer([z_mean, z_log_var])
    z = decode_dense_layer(sampling_z)
    z = layers.Reshape((H // 4, W // 4, C // 4))(z)
    recon_output = DecoderBlock2D(input_tensor=z, encoder=encoder,
                                  skip_connection_layer_names=skip_connect_layer_names, use_skip_connect=False,
                                  last_channel_num=input_shape[-1],
                                  block_size=block_size, groups=groups, num_downsample=num_downsample, padding=padding,
                                  base_act=base_act, last_act=last_act, name_prefix="recon")

    model = Model(input_tensor, recon_output)
    return model


def get_seg_multi_scale_model(input_shape, block_size=16,
                              num_class=6,
                              num_downsample=5,
                              base_act="relu", last_act="sigmoid"):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    split_shape = (input_shape[0],
                   input_shape[1],
                   input_shape[2] // 10)
    encoder_0, skip_connect_layer_names_0 = InceptionResNetV2(input_shape=split_shape, block_size=block_size, padding=padding,
                                                              groups=1, base_act=base_act, last_act=base_act, name_prefix="level_1",
                                                              skip_connect_names=True)
    encoder_1, skip_connect_layer_names_1 = InceptionResNetV2(input_shape=split_shape, block_size=block_size, padding=padding,
                                                              groups=1, base_act=base_act, last_act=base_act, name_prefix="level_0",
                                                              skip_connect_names=True)
    _, H, W, C = encoder_0.output.shape
    encoder_input_0 = encoder_0.input
    encoder_input_1 = encoder_1.input
    encoder_output_0 = encoder_0.output
    encoder_output_1 = encoder_1.output

    seg_output_0 = DecoderBlock2D(input_tensor=encoder_output_0, encoder=encoder_0,
                                  skip_connection_layer_names=skip_connect_layer_names_0, use_skip_connect=True,
                                  last_channel_num=num_class,
                                  groups=1, num_downsample=num_downsample,
                                  base_act=base_act, last_act=last_act, name_prefix="seg_0")
    seg_output_1 = DecoderBlock2D(input_tensor=encoder_output_1, encoder=encoder_1,
                                  skip_connection_layer_names=skip_connect_layer_names_1, use_skip_connect=True,
                                  last_channel_num=num_class,
                                  groups=1, num_downsample=num_downsample,
                                  base_act=base_act, last_act=last_act, name_prefix="seg_1")
    model_0 = Model(encoder_input_0, seg_output_0)
    model_1 = Model(encoder_input_1, seg_output_1)

    input_tensor = layers.Input(input_shape)
    input_tensor_list = tf.split(input_tensor, num_or_size_splits=10, axis=-1)

    output_tensor_list = []
    for idx, split_tensor in enumerate(input_tensor_list):
        print(idx)
        if idx < 9:
            output_tensor_list.append(model_0(split_tensor))
        else:
            output_tensor_list.append(model_1(split_tensor))
    output_tensor = layers.Concatenate(axis=-1)(output_tensor_list)
    return Model(input_tensor, output_tensor)


def get_seg_multi_scale_model(input_shape, block_size=16,
                              num_class=6,
                              num_downsample=5,
                              base_act="relu", last_act="sigmoid"):
    padding = "same"
    ################################################
    ################# Define Layer #################
    ################################################
    split_shape = (input_shape[0],
                   input_shape[1],
                   input_shape[2] // 10)
    encoder_0 = InceptionResNetV2_SkipConnectLevel_0(input_shape=split_shape, block_size=block_size, padding=padding,
                                                     groups=1, base_act=base_act, last_act=base_act, name_prefix="level_1")
    encoder_1 = InceptionResNetV2_SkipConnectLevel_1(input_shape=split_shape, block_size=block_size, padding=padding,
                                                     groups=1, base_act=base_act, last_act=base_act, name_prefix="level_0")
    input_tensor = layers.Input(input_shape)
    input_tensor_list = tf.split(input_tensor, num_or_size_splits=10, axis=-1)

    output_tensor_list_list = []
    for idx, split_tensor in enumerate(input_tensor_list):
        if idx < 9:
            output_tensor_list = encoder_0(split_tensor)
        else:
            output_tensor_list = encoder_1(split_tensor)
        output_tensor_list_list.append(output_tensor_list)
    level_0_tensor_recon_list = []
    level_0_tensor_concat_list_list = []
    level_1_tensor_concat_list = []
    # level_0 : [down_2, down_3, down_4, down_5, output]
    # level_1 : [down_1, down_2, down_3, down_4, down_5, output]
    # output_tensor_list_list: [level_1_0, level_1_1, ... level_1_8, level_0]
    for idx, level_0_output_tensor_list in enumerate(zip(*output_tensor_list_list[:9])):
        if idx == 0:
            continue
        level_0_recon = recon_overlapping_patches(
            level_0_output_tensor_list)
        level_0_tensor_recon_list.append(level_0_recon)
    # level_1_tensor_concat_list : [level_1_down_2,  level_1_down_3,  level_1_down_4, level_1_down_5, level_1_output]
    #                               level_0_recon_2, level_0_recon_3, level_0_recon_4, level_0_recon_5,
    for idx, level_1_output_tensor in enumerate(output_tensor_list_list[9]):
        level_1_concat_overlay = layers.Concatenate(axis=-1)([level_1_output_tensor,
                                                              level_0_tensor_recon_list[idx]])
        level_1_tensor_concat_list.append(level_1_concat_overlay)

    # level_0_tensor_concat_list : [level_1_down_1, level_1_down_2, level_1_down_3, level_1_down_4, level_1_down_5, level_1_output]
    #                               level_0_down_2, level_0_down_3, level_0_down_4, level_0_down_5,
    level_1_tensor_list_temp = output_tensor_list_list[9]
    for level_0_tensor_list_temp in output_tensor_list_list[:9]:
        level_0_tensor_concat_list_temp = []
        for idx, level_0_tensor_temp in enumerate(level_0_tensor_list_temp):
            if idx >= 4:
                level_0_tensor_concat_temp = level_0_tensor_temp
            else:
                level_1_tensor_temp = level_1_tensor_list_temp[idx]
                level_0_tensor_concat_temp = layers.Concatenate()([level_0_tensor_temp,
                                                                   level_1_tensor_temp])
            level_0_tensor_concat_list_temp.append(level_0_tensor_concat_temp)
        level_0_tensor_concat_list_list.append(level_0_tensor_concat_list_temp)

    seg_output_0_list = []
    for level_0_tensor_concat_list in level_0_tensor_concat_list_list:
        seg_output_0 = DecoderBlock2D_MultiScale(input_tensor=level_0_tensor_concat_list[-1],
                                                 skip_connect_tensor_list=level_0_tensor_concat_list[:-1],
                                                 last_channel_num=num_class,
                                                 groups=1,
                                                 base_act=base_act, last_act=last_act, name_prefix="seg_0")
        seg_output_0_list.append(seg_output_0)
    seg_output_0 = layers.Concatenate(axis=-1)(seg_output_0_list)
    seg_output_1 = DecoderBlock2D_MultiScale(input_tensor=level_1_tensor_concat_list[-1],
                                             skip_connect_tensor_list=level_1_tensor_concat_list[:-1],
                                             last_channel_num=num_class,
                                             groups=1,
                                             base_act=base_act, last_act=last_act, name_prefix="seg_0")
    seg_output = layers.Concatenate(axis=-1)([seg_output_0,
                                              seg_output_1])

    return Model(input_tensor, seg_output)


class MeanSTD(layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.flatten_layer = layers.Flatten()
        self.latent_dense_layer = layers.Dense(latent_dim * 4,
                                               activation=tf.nn.relu6)
        self.mean_dense_layer = layers.Dense(latent_dim, name="z_mean")
        self.log_var_dense_layer = layers.Dense(latent_dim, name="z_log_var")

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
