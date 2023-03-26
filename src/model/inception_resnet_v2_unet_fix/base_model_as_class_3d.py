from tensorflow.python.keras import backend
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from .layers import Pixelshuffle3D, get_norm_layer
from .transformer_layers import AddPositionEmbs
from .cbam_attention_module import attach_attention_module
from tensorflow_addons.layers import GroupNormalization, SpectralNormalization
CHANNEL_AXIS = -1
USE_CONV_BIAS = True
USE_DENSE_BIAS = True


def to_stride_tuple_3d(int_data):
    strides_tuple = (1, int_data, int_data, int_data, 1)
    return strides_tuple


def to_kernel_tuple_3d(int_or_list_or_tuple):
    if isinstance(int_or_list_or_tuple, int):
        int_or_list_or_tuple = (int_or_list_or_tuple,
                                int_or_list_or_tuple,
                                int_or_list_or_tuple)
    kernel_tuple = (int_or_list_or_tuple[0],
                    int_or_list_or_tuple[1],
                    int_or_list_or_tuple[2])
    return kernel_tuple


def to_pad_tuple_3d(int_or_list_or_tuple):
    if isinstance(int_or_list_or_tuple, int):
        int_or_list_or_tuple = (int_or_list_or_tuple,
                                int_or_list_or_tuple,
                                int_or_list_or_tuple)
    pad_tuple = (int_or_list_or_tuple[0] // 2,
                 int_or_list_or_tuple[1] // 2,
                 int_or_list_or_tuple[2] // 2)
    return pad_tuple


def downsize_hw(h, w):
    return h // 2, w // 2


def get_act_layer(activation, name=None):
    if activation is None:
        act_layer = layers.Lambda(lambda x: x, name=name)
    elif activation == 'relu':
        act_layer = layers.Activation(tf.nn.relu6, name=name)
    elif activation == 'leakyrelu':
        act_layer = layers.LeakyReLU(0.3, name=name)
    else:
        act_layer = layers.Activation(activation, name=name)
    return act_layer


def DecoderBlock3D(input_tensor=None,
                   encoder=None,
                   skip_connection_layer_names=None,
                   use_skip_connect=True,
                   last_channel_num=None,
                   groups=1,
                   num_downsample=5,
                   base_act="relu",
                   last_act="relu",
                   name_prefix=""):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    init_filter = encoder.output.shape[-1]
    x = Conv3DBN(init_filter, 3, groups=groups,
                 activation=base_act)(input_tensor)
    x = Conv3DBN(init_filter, 3, groups=groups,
                 activation=base_act)(x)

    for idx in range(num_downsample - 1, -1, -1):
        skip_connect = encoder.get_layer(
            skip_connection_layer_names[idx]).output
        filter_size = skip_connect.shape[-1]
        if use_skip_connect:
            x = layers.Concatenate(axis=-1)([x, skip_connect])
        x = Conv3DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        x = Conv3DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        x = Decoder3D(out_channel=filter_size, unsharp=True,
                      activation=base_act)(x)
        if idx == 0:
            x = OutputLayer3D(last_channel_num=last_channel_num,
                              act=last_act)(x)
    return x


def InceptionResNetV2_3D(input_shape=None,
                         block_size=16,
                         padding="valid",
                         groups=1,
                         base_act="relu",
                         last_act="relu",
                         name_prefix="",
                         use_attention=True,
                         skip_connect_names=False):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    input_tensor = layers.Input(input_shape)
    x = get_init_conv(input_tensor, block_size, groups,
                      base_act, 5, name_prefix)(input_tensor)
    x = get_block_1(x, block_size, padding,
                    groups, base_act, name_prefix)

    x = get_block_2(x, block_size, padding,
                    groups, base_act, name_prefix)
    x = get_block_3(x, block_size, padding,
                    groups, base_act, name_prefix)
    x = get_block_4(x, block_size, padding, groups,
                    use_attention, base_act, name_prefix)
    x = get_block_5(x, block_size, padding, groups,
                    use_attention, base_act, name_prefix)
    output_block = get_output_block(x, block_size, groups, use_attention,
                                    base_act, last_act, name_prefix)(x)

    model = Model(input_tensor, output_block,
                  name=f'{name_prefix}inception_resnet_v2')
    if skip_connect_names:
        skip_connect_name_list = [
            f"{name_prefix}down_block_{idx}" for idx in range(1, 6)]
        return model, skip_connect_name_list
    else:
        return model


def InceptionResNetV2_3D_Progressive(target_shape=None,
                                     block_size=16,
                                     padding="same",
                                     groups=1,
                                     norm="batch",
                                     base_act="relu",
                                     last_act="relu",
                                     name_prefix="",
                                     num_downsample=5,
                                     use_attention=True,
                                     skip_connect_names=False,
                                     small=False):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    final_downsample = 5

    input_shape = (target_shape[0] // (2 ** (final_downsample - num_downsample)),
                   target_shape[1] // (2 **
                                       (final_downsample - num_downsample)),
                   target_shape[2] // (2 **
                                       (final_downsample - num_downsample)),
                   target_shape[3])
    H, W, Z, _ = input_shape

    input_tensor = layers.Input(input_shape)
    x = get_init_conv(input_tensor, block_size, groups, norm, base_act,
                      num_downsample, name_prefix)(input_tensor)
    if num_downsample >= 5:
        x = get_block_1(x, block_size, padding,
                        groups, norm, base_act, name_prefix)
    if num_downsample >= 4:
        x = get_block_2(x, block_size, padding,
                        groups, norm, base_act, name_prefix)
    if num_downsample >= 3:
        x = get_block_3(x, block_size, padding,
                        groups, norm, base_act, name_prefix)
    if num_downsample >= 2:
        x = get_block_4(x, block_size, padding, groups,
                        use_attention, norm, base_act, name_prefix, small)
    if num_downsample >= 1:
        x = get_block_5(x, block_size, padding, groups,
                        use_attention, norm, base_act, name_prefix, small)
    x = get_output_block(x, block_size, padding, groups,
                         use_attention, norm, base_act, last_act, name_prefix, small)

    model = Model(input_tensor, x,
                  name=f'{name_prefix}inception_resnet_v2')
    if skip_connect_names:
        skip_connect_name_list = [
            f"{name_prefix}down_block_{idx}" for idx in range(1, 6)]
        return model, skip_connect_name_list
    else:
        return model


def get_init_conv(input_tensor, block_size, groups,
                  norm, activation, num_downsample, name_prefix):
    if num_downsample == 5:
        output_filter = block_size * 2
    elif num_downsample == 4:
        output_filter = block_size * 2
    elif num_downsample == 3:
        output_filter = block_size * 4
    elif num_downsample == 2:
        output_filter = block_size * 12
    elif num_downsample == 1:
        output_filter = block_size * 68
    model_input = layers.Input(input_tensor.shape[1:])
    conv = Conv3DBN(output_filter, 1, padding="same", groups=groups,
                    norm=norm, activation=activation)(model_input)
    return Model(model_input, conv,
                 name=f"{name_prefix}init_conv")


def get_block_1(input_tensor, block_size, padding,
                groups, norm, activation, name_prefix):
    conv = Conv3DBN(block_size * 2, 3, strides=2, padding=padding,
                    groups=groups, norm=norm, activation=activation,
                    name=f"{name_prefix}down_block_1")(input_tensor)
    return conv


def get_block_2(input_tensor, block_size, padding,
                groups, norm, activation, name_prefix):
    conv_1 = Conv3DBN(block_size * 2, 3, padding=padding, groups=groups,
                      norm=norm, activation=activation)(input_tensor)
    conv_2 = Conv3DBN(block_size * 4, 3, groups=groups,
                      norm=norm, activation=activation)(conv_1)
    max_pool = layers.MaxPooling3D(3, strides=2, padding=padding,
                                   name=f"{name_prefix}down_block_2")(conv_2)
    return max_pool


def get_block_3(input_tensor, block_size, padding,
                groups, norm, activation, name_prefix):
    conv_1 = Conv3DBN(block_size * 5, 1, padding=padding, groups=groups,
                      norm=norm, activation=activation)(input_tensor)
    conv_2 = Conv3DBN(block_size * 12, 3, padding=padding, groups=groups,
                      norm=norm, activation=activation)(conv_1)
    max_pool = layers.MaxPooling3D(3, strides=2, padding=padding,
                                   name=f"{name_prefix}down_block_3")(conv_2)
    return max_pool


def get_block_4(input_tensor, block_size, padding, groups,
                use_attention, norm, activation, name_prefix, small=False):
    branch_0 = Conv3DBN(block_size * 6, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv3DBN(block_size * 3, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv3DBN(block_size * 4, 5, groups=groups,
                        norm=norm, activation=activation)(branch_1)
    branch_2 = Conv3DBN(block_size * 4, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_2 = Conv3DBN(block_size * 6, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_2 = Conv3DBN(block_size * 6, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_pool = layers.AveragePooling3D(3, strides=1,
                                          padding='same')(input_tensor)
    branch_pool = Conv3DBN(block_size * 4, 1, groups=groups,
                           norm=norm, activation=activation)(branch_pool)
    branches_out = [branch_0, branch_1, branch_2, branch_pool]

    branches_out = layers.Concatenate(axis=CHANNEL_AXIS)(branches_out)
    if small:
        block_num = 3
    else:
        block_num = 10
    for idx in range(1, block_num + 1):
        branches_out = InceptionResnetBlock(scale=0.17,
                                            block_type='block35', block_size=block_size,
                                            groups=groups, norm=norm, activation=activation,
                                            use_attention=use_attention,
                                            name=f'{name_prefix}block_35_{idx}')(branches_out)
    return branches_out


def get_block_5(input_tensor, block_size, padding, groups,
                use_attention, norm, activation, name_prefix, small=False):
    if small:
        base_block_size = block_size * 2
    else:
        base_block_size = block_size * 8
    branch_0 = Conv3DBN(base_block_size * 3, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(input_tensor)
    branch_1 = Conv3DBN(base_block_size * 2, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv3DBN(base_block_size * 2, 3, groups=groups,
                        norm=norm, activation=activation)(branch_1)
    branch_1 = Conv3DBN(base_block_size * 3, 3, strides=2, padding=padding,
                        groups=groups, norm=norm, activation=activation)(branch_1)
    branch_pool = layers.AveragePooling3D(3, strides=2,
                                          padding=padding)(input_tensor)
    branches_output = [branch_0, branch_1, branch_pool]
    branches_output = layers.Concatenate(axis=CHANNEL_AXIS)(branches_output)

    if small:
        block_num = 5
    else:
        block_num = 10
    for idx in range(1, block_num + 1):
        if idx == block_num:
            block_name = f"{name_prefix}block_5"
        else:
            block_name = f'{name_prefix}block_17_{idx}'
        branches_output = InceptionResnetBlock(scale=0.11,
                                               block_type='block17', block_size=block_size,
                                               groups=groups, norm=norm, activation=activation,
                                               use_attention=use_attention,
                                               name=block_name,
                                               small=small)(branches_output)
    return branches_output


def get_output_block(input_tensor, block_size, padding, groups, use_attention,
                     norm, activation, last_activation, name_prefix, small):

    if small:
        base_block_size = block_size * 2
    else:
        base_block_size = block_size * 8

    branch_0 = Conv3DBN(base_block_size * 2, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_0 = Conv3DBN(base_block_size * 3, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(branch_0)
    branch_1 = Conv3DBN(base_block_size * 2, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv3DBN(base_block_size * 2, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(branch_1)
    branch_2 = Conv3DBN(base_block_size * 2, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_2 = Conv3DBN(base_block_size * 2, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_2 = Conv3DBN(base_block_size * 3, 3, strides=2, padding=padding,
                        groups=groups, norm=norm, activation=activation)(branch_2)
    branch_pool = layers.MaxPooling3D(3, strides=2,
                                      padding=padding)(input_tensor)
    branches_output = [branch_0, branch_1, branch_2, branch_pool]
    branches_output = layers.Concatenate(axis=CHANNEL_AXIS)(branches_output)
    if small:
        block_num = 3
    else:
        block_num = 10
    for idx in range(1, block_num + 1):
        branches_output = InceptionResnetBlock(scale=0.2,
                                               block_type='block8', block_size=block_size,
                                               groups=groups, norm=norm, activation=activation,
                                               use_attention=use_attention,
                                               name=f'{name_prefix}block_8_{idx}',
                                               small=small)(branches_output)
    branches_output = Conv3DBN(base_block_size * 12, 1, groups=groups,
                               norm=norm, activation=last_activation)(branches_output)
    return branches_output


class Conv3DBN(layers.Layer):
    def __init__(self, filters, kernel_size,
                 strides=1, padding="same", groups=1,
                 norm="batch", activation="relu", use_bias=False, name=None):
        super().__init__()
        norm_axis = CHANNEL_AXIS
        self.norm = norm
        if norm == "spectral":
            self.conv_layer = SpectralNormalization(layers.Conv3D(filters=filters,
                                                                  kernel_size=kernel_size,
                                                                  strides=strides,
                                                                  padding=padding,
                                                                  groups=groups,
                                                                  use_bias=use_bias,
                                                                  kernel_initializer='glorot_uniform',
                                                                  bias_initializer='zeros'))
        else:
            if groups == 1:
                # self.conv_layer = EqualizedConv3D(out_channels=filters,
                #                                   kernel=kernel_size,
                #                                   downsample=strides == 2,
                #                                   padding=padding,
                #                                   use_bias=use_bias)
                self.conv_layer = layers.Conv3D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                groups=groups,
                                                use_bias=use_bias,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros')
                if use_bias:
                    self.norm_layer = get_norm_layer(None)
                else:
                    self.norm_layer = get_norm_layer(norm, axis=norm_axis)
            else:
                self.conv_layer = layers.Conv3D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                groups=groups,
                                                use_bias=use_bias,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros')
                self.norm_layer = GroupNormalization(groups=groups,
                                                     axis=norm_axis,
                                                     scale=False)
        self.act_layer = get_act_layer(activation, name=name)

    def __call__(self, input_tensor):
        output = self.conv_layer(input_tensor)
        if self.norm == "spectral":
            pass
        else:
            output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class EqualizedConv3D(layers.Layer):
    def __init__(self, out_channels, downsample=False, kernel=3,
                 padding="same", use_bias=True, gain=2, **kwargs):
        super().__init__(**kwargs)
        if downsample:
            self.strides = to_stride_tuple_3d(2)
        else:
            self.strides = to_stride_tuple_3d(1)
        self.kernel = to_kernel_tuple_3d(kernel)
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.gain = gain
        if padding == "same":
            self.pad = to_pad_tuple_3d(kernel)
        elif padding == "valid":
            self.pad = False

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel[0], self.kernel[1], self.kernel[2],
                   self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
            )
        fan_in = self.kernel[0] * self.kernel[1] * \
            self.kernel[2] * self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs):
        if isinstance(self.pad, tuple):
            x = tf.pad(inputs, [
                [0, 0],
                [self.pad[0], self.pad[0]],
                [self.pad[1], self.pad[1]],
                [self.pad[2], self.pad[2]],
                [0, 0]
            ], mode="REFLECT")
        else:
            x = inputs
        output = tf.nn.conv3d(x, self.scale * self.w, strides=self.strides,
                              padding="VALID")
        if self.use_bias:
            output = output + self.b

        return output


class ConcatBlock(layers.Layer):
    def __init__(self, layer_list):
        super().__init__()
        concat_axis = CHANNEL_AXIS
        self.layer_list = layer_list
        self.concat_layer = layers.Concatenate(axis=concat_axis)

    def call(self, input_tensor):
        tensor_list = [layer(input_tensor) for layer in self.layer_list]
        concat = self.concat_layer(tensor_list)
        return concat


class InceptionResnetBlock(layers.Layer):
    def __init__(self, scale, block_type,
                 block_size=16, groups=1,
                 norm="batch", activation='relu',
                 use_attention=True, name=None, small=False):
        super().__init__(name=name)
        self.use_attention = use_attention
        if small:
            block_size = block_size // 4
        if block_type == 'block35':
            branch_0 = Conv3DBN(block_size * 2, 1, groups=groups,
                                norm=norm, activation=activation)
            branch_1_1 = Conv3DBN(block_size * 2, 1, groups=groups,
                                  norm=norm, activation=activation)
            branch_1_2 = Conv3DBN(block_size * 2, 3, groups=groups,
                                  norm=norm, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2])
            branch_2_1 = Conv3DBN(block_size * 2, 1, groups=groups,
                                  norm=norm, activation=activation)
            branch_2_2 = Conv3DBN(block_size * 3, 3, groups=groups,
                                  norm=norm, activation=activation)
            branch_2_3 = Conv3DBN(block_size * 4, 3, groups=groups,
                                  norm=norm, activation=activation)
            branch_2 = Sequential([branch_2_1,
                                   branch_2_2,
                                   branch_2_3])
            branches = [branch_0, branch_1, branch_2]
            up_channel = block_size * 20
        elif block_type == 'block17':
            branch_0 = Conv3DBN(block_size * 12, 1, groups=groups,
                                norm=norm, activation=activation)
            branch_1_1 = Conv3DBN(block_size * 8, 1, groups=groups,
                                  norm=norm, activation=activation)
            branch_1_2 = Conv3DBN(block_size * 10, [1, 1, 7], groups=groups,
                                  norm=norm, activation=activation)
            branch_1_3 = Conv3DBN(block_size * 10, [1, 7, 1], groups=groups,
                                  norm=norm, activation=activation)
            branch_1_4 = Conv3DBN(block_size * 10, [7, 1, 1], groups=groups,
                                  norm=norm, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3,
                                   branch_1_4])
            branches = [branch_0, branch_1]
            up_channel = block_size * 128
        elif block_type == 'block8':
            branch_0 = Conv3DBN(block_size * 12, 1, groups=groups,
                                norm=norm, activation=activation)
            branch_1_1 = Conv3DBN(block_size * 12, 1, groups=groups,
                                  norm=norm, activation=activation)
            branch_1_2 = Conv3DBN(block_size * 14, [1, 1, 3], groups=groups,
                                  norm=norm, activation=activation)
            branch_1_3 = Conv3DBN(block_size * 16, [1, 3, 1], groups=groups,
                                  norm=norm, activation=activation)
            branch_1_4 = Conv3DBN(block_size * 16, [3, 1, 1], groups=groups,
                                  norm=norm, activation=activation)

            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3,
                                   branch_1_4])
            branches = [branch_0, branch_1]
            up_channel = block_size * 192
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.branch_layer = ConcatBlock(branches)
        self.up_layer = Conv3DBN(up_channel, 1, groups=groups,
                                 norm=norm, activation=None, use_bias=True)
        if self.use_attention:
            self.attention_layer = CBAM_Block3D(up_channel, ratio=8)
        self.residual_block = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=up_channel,
            arguments={'scale': scale})
        self.act_layer = get_act_layer(activation)

    def call(self, input_tensor):
        branch = self.branch_layer(input_tensor)
        up = self.up_layer(branch)
        if self.use_attention:
            up = self.attention_layer(up)
        residual = self.residual_block([input_tensor, up])
        act = self.act_layer(residual)
        return act


class CBAM_Block3D(layers.Layer):
    def __init__(self, input_filter, ratio=8):
        super().__init__()
        self.channel_attention_layer = ChannelAttention3D(input_filter,
                                                          ratio=ratio)
        self.spatial_attention_layer = SpatialAttention3D()

    def call(self, input_tensor):
        channel_attention = self.channel_attention_layer(input_tensor)
        spatial_attention = self.spatial_attention_layer(channel_attention)
        return spatial_attention


class ChannelAttention3D(layers.Layer):
    def __init__(self, input_filter, ratio=8):
        super().__init__()
        self.shared_dense_one = layers.Dense(input_filter // ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_dense_two = layers.Dense(input_filter,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.avg_pool_layer = layers.GlobalAveragePooling3D()
        self.max_pool_layer = layers.GlobalMaxPooling3D()
        self.reshape_layer = layers.Reshape((1, 1, 1, input_filter))
        self.act_layer = get_act_layer('sigmoid')

    def call(self, input_tensor):

        avg_pool = self.avg_pool_layer(input_tensor)
        avg_pool = self.reshape_layer(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.max_pool_layer(input_tensor)
        max_pool = self.reshape_layer(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        cbam_feature = avg_pool + max_pool
        cbam_feature = self.act_layer(cbam_feature)
        output = input_tensor * cbam_feature
        return output


class SpatialAttention3D(layers.Layer):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.avg_pool_layer = layers.Lambda(
            lambda x: backend.mean(x, axis=-1, keepdims=True))
        self.max_pool_layer = layers.Lambda(
            lambda x: backend.max(x, axis=-1, keepdims=True))
        self.concat_layer = layers.Concatenate(axis=-1)
        self.cbam_conv_layer = layers.Conv3D(filters=1,
                                             kernel_size=kernel_size,
                                             strides=1,
                                             padding='same',
                                             activation='sigmoid',
                                             kernel_initializer='he_normal',
                                             use_bias=False)

    def call(self, input_tensor):
        avg_pool = self.avg_pool_layer(input_tensor)
        max_pool = self.max_pool_layer(input_tensor)
        concat = self.concat_layer([avg_pool, max_pool])
        cbam_feature = self.cbam_conv_layer(concat)
        output = input_tensor * cbam_feature
        return output


class Decoder3D(layers.Layer):
    def __init__(self, out_channel, kernel_size=2, groups=1,
                 activation="tanh"):
        super().__init__()
        USE_CONV_BIAS = True
        self.kernel_size = kernel_size
        self.conv_before_pixel_shffle = layers.Conv3D(filters=out_channel * (kernel_size ** 2),
                                                      kernel_size=1, padding="same",
                                                      strides=1, use_bias=USE_CONV_BIAS)
        self.pixel_suffle = Pixelshuffle3D(kernel_size=2)
        self.conv_after_pixel_shffle = layers.Conv3D(filters=out_channel,
                                                     kernel_size=1, padding="same",
                                                     strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv3D(filters=out_channel,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)

        self.upsample_layer = layers.UpSampling3D(size=kernel_size,
                                                  interpolation="bilinear")
        self.conv_after_upsample = layers.Conv3D(filters=out_channel,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)
        self.norm_layer_pixel_shffle = GroupNormalization(groups=groups,
                                                          axis=-1)
        self.norm_layer_upsample = GroupNormalization(groups=groups,
                                                      axis=-1)
        self.act_layer = get_act_layer(activation)

    def call(self, x):

        pixel_shuffle = self.conv_before_pixel_shffle(x)
        pixel_shuffle = self.pixel_suffle(pixel_shuffle)
        pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)
        pixel_shuffle = self.norm_layer_pixel_shffle(pixel_shuffle)

        upsample = self.conv_before_upsample(x)
        upsample = self.upsample_layer(upsample)
        upsample = self.conv_after_upsample(upsample)
        upsample = self.norm_layer_upsample(upsample)

        output = layers.Concatenate()([pixel_shuffle, upsample])
        output = self.act_layer(output)
        return output


class OutputLayer3D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        USE_CONV_BIAS = True
        self.conv_1x1 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=1,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.conv_3x3 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.act = get_act_layer(act)

    def call(self, input_tensor):
        conv_1x1 = self.conv_1x1(input_tensor)
        conv_3x3 = self.conv_3x3(input_tensor)
        output = (conv_1x1 + conv_3x3) / 2
        output = self.act(output)

        return output
