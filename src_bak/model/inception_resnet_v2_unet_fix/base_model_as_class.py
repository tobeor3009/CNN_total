from tensorflow.python.keras import backend
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from .layers import EqualizedConv, EqualizedDense, UnsharpMasking2D, get_norm_layer, SimpleOutputLayer2D
from .layers import OutputLayer2D as HighwayOutputLayer2D, PixelShuffleBlock2D
from .layers import UpsampleBlock2D, UpsampleBlock2DHeatmap, Decoder2D, Decoder2DHeatmap, TransformerEncoder
from .transformer_layers import AddPositionEmbs
from .cbam_attention_module import attach_attention_module
from tensorflow_addons.layers import GroupNormalization, SpectralNormalization
CHANNEL_AXIS = -1
USE_CONV_BIAS = True
USE_DENSE_BIAS = True


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


def DecoderBlock2D(input_tensor=None,
                   encoder=None,
                   skip_connection_layer_names=None,
                   use_skip_connect=True,
                   last_channel_num=None,
                   groups=1,
                   num_downsample=5,
                   norm="batch",
                   base_act="relu",
                   last_act="relu",
                   name_prefix=""):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    init_filter = encoder.output.shape[-1]
    x = Conv2DBN(init_filter, 3, groups=groups,
                 activation=base_act)(input_tensor)
    x = Conv2DBN(init_filter, 3, groups=groups,
                 activation=base_act)(x)

    for idx in range(num_downsample - 1, -1, -1):
        skip_connect = encoder.get_layer(
            skip_connection_layer_names[idx]).output
        filter_size = skip_connect.shape[-1]
        x = Conv2DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        x = Conv2DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        if idx % 2 == 0:
            x = UpsampleBlock2D(out_channel=filter_size, unsharp=False,
                                norm=norm, activation=base_act)(x)
        else:
            x = PixelShuffleBlock2D(out_channel=filter_size, unsharp=False,
                                    norm=norm, activation=base_act)(x)
        if idx == 0:
            x = SimpleOutputLayer2D(last_channel_num=last_channel_num,
                                    act=last_act)(x)
        else:
            if use_skip_connect:
                x = layers.Concatenate(axis=-1)([x, skip_connect])
    return x


def DecoderBlock2D_stargan(input_tensor=None,
                           skip_connect_tensor_list=None,
                           last_channel_num=None,
                           filter_scale=1,
                           groups=1,
                           num_downsample=5,
                           norm="batch",
                           base_act="relu",
                           last_act="relu",
                           name_prefix=""):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    init_filter = input_tensor.shape[-1] // 2
    x = Conv2DBN(init_filter, 3, groups=groups,
                 norm=norm, activation=base_act)(input_tensor)
    x = Conv2DBN(init_filter, 3, groups=groups,
                 norm=norm, activation=base_act)(x)

    for idx in range(num_downsample - 1, -1, -1):
        skip_connect = skip_connect_tensor_list[idx]
        filter_size = int(round(skip_connect.shape[-1] * filter_scale))

        x = Conv2DBN(filter_size, 3, groups=groups,
                     norm=norm, activation=base_act)(x)
        x = Conv2DBN(filter_size, 3, groups=groups,
                     norm=norm, activation=base_act)(x)

        if idx % 2 == 0:
            x = UpsampleBlock2D(out_channel=filter_size, unsharp=False,
                                norm=norm, activation=base_act)(x)
        else:
            x = PixelShuffleBlock2D(out_channel=filter_size, unsharp=False,
                                    norm=norm, activation=base_act)(x)

        if idx == 0:
            x = SimpleOutputLayer2D(last_channel_num=last_channel_num,
                                    act=last_act)(x)
        else:
            x = layers.Concatenate(axis=-1)([x, skip_connect])
    return x


def DecoderBlock2D_stargan_heatmap(input_tensor=None,
                                   skip_connect_tensor_list=None,
                                   heatmap_list=None,
                                   last_channel_num=None,
                                   filter_scale=1,
                                   groups=1,
                                   num_downsample=5,
                                   norm="batch",
                                   base_act="relu",
                                   last_act="relu",
                                   name_prefix=""):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    init_filter = input_tensor.shape[-1] // 2
    x = Conv2DBN(init_filter, 3, groups=groups,
                 norm=norm, activation=base_act)(input_tensor)
    x = Conv2DBN(init_filter, 3, groups=groups,
                 norm=norm, activation=base_act)(x)

    for idx in range(num_downsample - 1, -1, -1):
        skip_connect = skip_connect_tensor_list[idx]
        heatmap = heatmap_list[idx]
        filter_size = int(round(skip_connect.shape[-1] * filter_scale))

        x = Conv2DBN(filter_size, 3, groups=groups,
                     norm=norm, activation=base_act)(x)
        x = Conv2DBN(filter_size, 3, groups=groups,
                     norm=norm, activation=base_act)(x)

        if idx == 0:
            x = Decoder2DHeatmap(out_channel=filter_size, unsharp=False,
                                 norm=norm, activation=base_act)(x, heatmap)
            x = layers.Concatenate(axis=-1)([x, skip_connect])
            x = SimpleOutputLayer2D(last_channel_num=last_channel_num,
                                    act=last_act)(x)
        else:
            x = UpsampleBlock2DHeatmap(out_channel=filter_size, unsharp=False,
                                       norm=norm, activation=base_act)(x, heatmap)
            x = layers.Concatenate(axis=-1)([x, skip_connect])
    return x


def DecoderBlock2D_MultiScale(input_tensor=None,
                              skip_connect_tensor_list=None,
                              last_channel_num=None,
                              groups=1,
                              base_act="relu",
                              last_act="relu",
                              name_prefix=""):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    model_input = layers.Input(input_tensor.shape[1:])
    init_filter = input_tensor.shape[-1] // 2
    x = Conv2DBN(init_filter, 3, groups=groups,
                 activation=base_act)(model_input)
    x = Conv2DBN(init_filter, 3, groups=groups,
                 activation=base_act)(x)
    skip_connect_input_list = []
    for idx in range(len(skip_connect_tensor_list) - 1, -1, -1):
        skip_connect_refer = skip_connect_tensor_list[idx]
        skip_connect_input = layers.Input(skip_connect_refer.shape[1:])
        skip_connect_input_list.insert(0, skip_connect_input)
        filter_size = skip_connect_refer.shape[-1]
        x = Conv2DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        x = Conv2DBN(filter_size, 3, groups=groups,
                     activation=base_act)(x)
        x = Decoder2D(out_channel=filter_size, kernel_size=2,
                      unsharp=False, activation=base_act)(x)
        x = layers.Concatenate(axis=-1)([x, skip_connect_input])
        if idx == 0:
            x = OutputLayer2D(last_channel_num=last_channel_num,
                              act=last_act)(x)

    return Model([*skip_connect_input_list, model_input], x)


def InceptionResNetV2(input_shape=None,
                      block_size=16,
                      padding="valid",
                      pooling="max",
                      groups=1,
                      norm="batch",
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
    num_downsample = 5
    x = get_block_1(input_tensor, block_size, groups,
                    norm, base_act, num_downsample, name_prefix)
    x = get_block_2(x, block_size, padding, groups,
                    norm, base_act, name_prefix)
    x = get_block_3(x, block_size, padding, pooling,
                    groups, norm, base_act, name_prefix)
    x = get_block_4(x, block_size, padding, pooling, groups,
                    use_attention, norm, base_act, name_prefix)
    x = get_block_5(x, block_size, padding, groups,
                    use_attention, norm, base_act, name_prefix)
    output_block = get_output_block(x, block_size, padding, groups,
                                    use_attention, norm, base_act, last_act, name_prefix)

    model = Model(input_tensor, output_block,
                  name=f'{name_prefix}inception_resnet_v2')
    if skip_connect_names:
        skip_connect_name_list = [
            f"{name_prefix}block_{idx}" for idx in range(1, 6)]
        return model, skip_connect_name_list
    else:
        return model


def InceptionResNetV2_SkipConnect(input_shape=None,
                                  block_size=16,
                                  padding="valid",
                                  pooling="max",
                                  groups=1,
                                  norm="batch",
                                  base_act="relu",
                                  last_act="relu",
                                  name_prefix="",
                                  use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    input_tensor = layers.Input(input_shape)
    num_downsample = 5
    down_1 = get_block_1(input_tensor, block_size, groups,
                         norm, base_act, num_downsample, name_prefix)
    down_2 = get_block_2(down_1, block_size, padding, groups,
                         norm, base_act, name_prefix)
    down_3 = get_block_3(down_2, block_size, padding, pooling,
                         groups, norm, base_act, name_prefix)
    down_4 = get_block_4(down_3, block_size, padding, pooling, groups,
                         use_attention, norm, base_act, name_prefix)
    down_5 = get_block_5(down_4, block_size, padding, groups,
                         use_attention, norm, base_act, name_prefix)
    output_block = get_output_block(down_5, block_size, padding, groups,
                                    use_attention, norm, base_act, last_act, name_prefix)

    model = Model(input_tensor,
                  [down_1, down_2, down_3, down_4, down_5, output_block],
                  name=f'{name_prefix}inception_resnet_v2')

    return model


def InceptionResNetV2_SkipConnectLevel_0(input_shape=None,
                                         block_size=16,
                                         padding="valid",
                                         pooling="max",
                                         groups=1,
                                         norm="batch",
                                         base_act="relu",
                                         last_act="relu",
                                         name_prefix="",
                                         use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    input_tensor = layers.Input(input_shape)
    num_downsample = 5
    down_1 = get_block_1(input_tensor, block_size, groups,
                         norm, base_act, num_downsample, name_prefix)
    down_2 = get_block_2(down_1, block_size, padding, groups,
                         norm, base_act, name_prefix)
    down_3 = get_block_3(down_2, block_size, padding, pooling,
                         groups, norm, base_act, name_prefix)
    down_4 = get_block_4(down_3, block_size, padding, pooling, groups,
                         use_attention, norm, base_act, name_prefix)
    down_5 = get_block_5(down_4, block_size, padding, groups,
                         use_attention, norm, base_act, name_prefix)
    output_block = get_output_block(down_5, block_size, padding, groups,
                                    use_attention, norm, base_act, last_act, name_prefix)

    model = Model(input_tensor,
                  [down_1, down_2, down_3, down_4, down_5, output_block],
                  name=f'{name_prefix}inception_resnet_v2')

    return model


def InceptionResNetV2_SkipConnectLevel_1(input_shape=None,
                                         block_size=16,
                                         padding="valid",
                                         pooling="max",
                                         groups=1,
                                         norm="batch",
                                         base_act="relu",
                                         last_act="relu",
                                         name_prefix="",
                                         use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    num_downsample = 4
    input_tensor = layers.Input(input_shape)
    down_2 = get_block_1(input_tensor, block_size, groups,
                         norm, base_act, num_downsample, name_prefix)
    down_3 = get_block_3(down_2, block_size, padding, pooling,
                         groups, norm, base_act, name_prefix)
    down_4 = get_block_4(down_3, block_size, padding, pooling, groups,
                         use_attention, norm, base_act, name_prefix)
    down_5 = get_block_5(down_4, block_size, padding, groups,
                         use_attention, norm, base_act, name_prefix)
    output_block = get_output_block(down_5, block_size, padding, groups,
                                    use_attention, norm, base_act, last_act, name_prefix)

    model = Model(input_tensor,
                  [down_2, down_3, down_4, down_5, output_block],
                  name=f'{name_prefix}inception_resnet_v2')

    return model


def InceptionResNetV2_Stargan(input_shape=None,
                              label_len=None,
                              block_size=16,
                              decoder_filter_scale=1,
                              padding="same",
                              pooling="max",
                              groups=1,
                              norm="batch",
                              base_act="relu",
                              last_act="relu",
                              last_channel_num=1,
                              name_prefix="",
                              use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    skip_connect_tensor_list = []
    input_tensor = layers.Input(input_shape)
    class_input = layers.Input(shape=(label_len))
    num_downsample = 5
    x = get_block_1(input_tensor, block_size, groups,
                    norm, base_act, num_downsample, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_2(x, block_size, padding, groups,
                    norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_3(x, block_size, padding, pooling,
                    groups, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_4(x, block_size, padding, pooling, groups,
                    use_attention, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_5(x, block_size, padding, groups,
                    use_attention, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_output_block(x, block_size, padding, groups,
                         use_attention, norm, base_act, last_act, name_prefix)
    _, H, W, C = backend.int_shape(x)

    class_tensor = layers.Reshape((1, 1, label_len))(class_input)
    class_tensor = tf.tile(class_tensor, (1, H, W, 1))

    x = layers.Concatenate(axis=-1)([x, class_tensor])
    x = DecoderBlock2D_stargan(input_tensor=x,
                               skip_connect_tensor_list=skip_connect_tensor_list,
                               last_channel_num=last_channel_num,
                               filter_scale=decoder_filter_scale,
                               groups=1,
                               num_downsample=5,
                               norm=norm,
                               base_act=base_act,
                               last_act=last_act)
    return Model([input_tensor, class_input], x)


def InceptionResNetV2_Stargan_Progressive(input_shape=None,
                                          label_len=None,
                                          block_size=16,
                                          decoder_filter_scale=1,
                                          num_downsample=5,
                                          padding="same",
                                          pooling="max",
                                          groups=1,
                                          norm="batch",
                                          base_act="relu",
                                          last_act="relu",
                                          last_channel_num=1,
                                          name_prefix="",
                                          use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    skip_connect_tensor_list = []
    input_tensor = layers.Input(input_shape)
    class_input = layers.Input(shape=(label_len))

    x = get_block_1(input_tensor, block_size, groups,
                    norm, base_act, num_downsample, name_prefix)
    skip_connect_tensor_list.append(x)
    if num_downsample >= 5:
        x = get_block_2(x, block_size, padding, groups,
                        norm, base_act, name_prefix)
        skip_connect_tensor_list.append(x)
    if num_downsample >= 4:
        x = get_block_3(x, block_size, padding, pooling,
                        groups, norm, base_act, name_prefix)
        skip_connect_tensor_list.append(x)
    if num_downsample >= 3:
        x = get_block_4(x, block_size, padding, pooling, groups,
                        use_attention, norm, base_act, name_prefix)
        skip_connect_tensor_list.append(x)
    if num_downsample >= 2:
        x = get_block_5(x, block_size, padding, groups,
                        use_attention, norm, base_act, name_prefix)
        skip_connect_tensor_list.append(x)
    if num_downsample >= 1:
        x = get_output_block(x, block_size, padding, groups,
                             use_attention, norm, base_act, last_act, name_prefix)
    _, H, W, C = backend.int_shape(x)

    class_tensor = layers.Reshape((1, 1, label_len))(class_input)
    class_tensor = tf.tile(class_tensor, (1, H, W, 1))

    x = layers.Concatenate(axis=-1)([x, class_tensor])
    x = DecoderBlock2D_stargan(input_tensor=x,
                               skip_connect_tensor_list=skip_connect_tensor_list,
                               last_channel_num=last_channel_num,
                               filter_scale=decoder_filter_scale,
                               groups=1,
                               num_downsample=num_downsample,
                               norm=norm,
                               base_act=base_act,
                               last_act=last_act)
    return Model([input_tensor, class_input], x)


def InceptionResNetV2_StarganHeatmap(input_shape=None,
                                     label_len=None,
                                     block_size=16,
                                     decoder_filter_scale=1,
                                     padding="same",
                                     pooling="max",
                                     groups=1,
                                     norm="batch",
                                     base_act="relu",
                                     last_act="relu",
                                     last_channel_num=1,
                                     name_prefix="",
                                     use_attention=True):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    skip_connect_tensor_list = []
    input_tensor = layers.Input(input_shape)
    class_input = layers.Input(shape=(label_len))
    num_downsample = 5
    x = get_block_1(input_tensor, block_size, groups,
                    norm, base_act, num_downsample, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_2(x, block_size, padding, groups,
                    norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_3(x, block_size, padding, pooling,
                    groups, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_4(x, block_size, padding, pooling, groups,
                    use_attention, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_block_5(x, block_size, padding, groups,
                    use_attention, norm, base_act, name_prefix)
    skip_connect_tensor_list.append(x)
    x = get_output_block(x, block_size, padding, groups,
                         use_attention, norm, base_act, last_act, name_prefix)
    _, h, w, _ = backend.int_shape(x)
    class_tensor = layers.Reshape((1, 1, label_len))(class_input)
    class_tensor = tf.tile(class_tensor, (1, h, w, 1))

    x = layers.Concatenate(axis=-1)([x, class_tensor])

    heatmap_input_list = []
    for _ in skip_connect_tensor_list:
        heatmap_input_list.append(layers.Input((None, None, None)))
    x = DecoderBlock2D_stargan_heatmap(input_tensor=x,
                                       skip_connect_tensor_list=skip_connect_tensor_list,
                                       heatmap_list=heatmap_input_list,
                                       last_channel_num=last_channel_num,
                                       filter_scale=decoder_filter_scale,
                                       groups=1,
                                       num_downsample=5,
                                       norm=norm,
                                       base_act=base_act,
                                       last_act=last_act)
    return Model([input_tensor, class_input, *heatmap_input_list], x)


def InceptionResNetV2_progressive(target_shape=None,
                                  block_size=16,
                                  padding="same",
                                  pooling="average",
                                  groups=1,
                                  norm="batch",
                                  base_act="relu",
                                  last_act="relu",
                                  name_prefix="",
                                  num_downsample=5,
                                  use_attention=True,
                                  skip_connect_tensor=False,
                                  skip_connect_names=False):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"

    final_downsample = 5

    input_shape = (target_shape[0] // (2 ** (final_downsample - num_downsample)),
                   target_shape[1] // (2 **
                                       (final_downsample - num_downsample)),
                   target_shape[2])
    H, W, _ = input_shape

    input_tensor = layers.Input(input_shape)
    skip_connect_list = []
    x = get_block_1(input_tensor, block_size, groups,
                    norm, base_act, num_downsample, name_prefix)
    skip_connect_list.append(x)
    if num_downsample >= 5:
        x = get_block_2(x, block_size, padding, groups,
                        norm, base_act, name_prefix)
        skip_connect_list.append(x)
    if num_downsample >= 4:
        x = get_block_3(x, block_size, padding, pooling,
                        groups, norm, base_act, name_prefix)
        skip_connect_list.append(x)
    if num_downsample >= 3:
        x = get_block_4(x, block_size, padding, pooling, groups,
                        use_attention, norm, base_act, name_prefix)
        skip_connect_list.append(x)
    if num_downsample >= 2:
        x = get_block_5(x, block_size, padding, groups,
                        use_attention, norm, base_act, name_prefix)
        skip_connect_list.append(x)
    if num_downsample >= 1:
        x = get_output_block(x, block_size, padding, groups,
                             use_attention, norm, base_act, last_act, name_prefix)
    if skip_connect_tensor:
        model = Model(input_tensor, [x, skip_connect_list],
                      name=f'{name_prefix}inception_resnet_v2')
    else:
        model = Model(input_tensor, x,
                      name=f'{name_prefix}inception_resnet_v2')
    if skip_connect_names:
        skip_connect_name_list = [f"{name_prefix}block_1"] + \
            [f"{name_prefix}block_{idx}"
             for idx in range(6 - num_downsample + 1, 6)]

        return model, skip_connect_name_list
    else:
        return model


def get_block_1(input_tensor, block_size, groups,
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
    conv = Conv2DBN(output_filter, 1, padding="same", groups=groups,
                    norm=norm, activation=activation,
                    name=f"{name_prefix}block_1")(input_tensor)
    return conv


def get_block_2(input_tensor, block_size, padding,
                groups, norm, activation, name_prefix):
    conv_down = Conv2DBN(block_size * 2, 3, strides=2, padding=padding,
                         groups=groups, norm=norm, activation=activation)(input_tensor)
    conv_1 = Conv2DBN(block_size * 2, 3, padding=padding, groups=groups,
                      norm=norm, activation=activation)(conv_down)
    conv_2 = Conv2DBN(block_size * 4, 3, groups=groups,
                      norm=norm, activation=activation,
                      name=f"{name_prefix}block_2")(conv_1)

    return conv_2


def get_block_3(input_tensor, block_size, padding, pooling,
                groups, norm, activation, name_prefix):
    if pooling == "average":
        pool = layers.AveragePooling2D(3, strides=2, padding=padding,
                                       )(input_tensor)
    elif pooling == "max":
        pool = layers.MaxPooling2D(3, strides=2, padding=padding,
                                   )(input_tensor)
    conv_1 = Conv2DBN(block_size * 5, 1, padding=padding, groups=groups,
                      norm=norm, activation=activation)(pool)
    conv_2 = Conv2DBN(block_size * 12, 3, padding=padding, groups=groups,
                      norm=norm, activation=activation,
                      name=f"{name_prefix}block_3")(conv_1)
    return conv_2


def get_block_4(input_tensor, block_size, padding, pooling,
                groups, use_attention, norm, activation, name_prefix):

    if pooling == "average":
        pool = layers.AveragePooling2D(
            3, strides=2, padding=padding)(input_tensor)
    elif pooling == "max":
        pool = layers.MaxPooling2D(3, strides=2, padding=padding)(input_tensor)
    branch_0 = Conv2DBN(block_size * 6, 1, groups=groups,
                        norm=norm, activation=activation)(pool)
    branch_1 = Conv2DBN(block_size * 3, 1, groups=groups,
                        norm=norm, activation=activation)(pool)
    branch_1 = Conv2DBN(block_size * 4, 5, groups=groups,
                        norm=norm, activation=activation)(branch_1)
    branch_2 = Conv2DBN(block_size * 4, 1, groups=groups,
                        norm=norm, activation=activation)(pool)
    branch_2 = Conv2DBN(block_size * 6, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_2 = Conv2DBN(block_size * 6, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_pool = layers.AveragePooling2D(3, strides=1,
                                          padding='same')(pool)
    branch_pool = Conv2DBN(block_size * 4, 1, groups=groups,
                           norm=norm, activation=activation)(branch_pool)
    branches_out = [branch_0, branch_1, branch_2, branch_pool]

    branches_out = layers.Concatenate(axis=CHANNEL_AXIS)(branches_out)

    for idx in range(1, 11):
        if idx == 10:
            block_name = f"{name_prefix}block_4"
        else:
            block_name = f'{name_prefix}block_35_{idx}'
        branches_out = InceptionResnetBlock(scale=0.17,
                                            block_type='block35', block_size=block_size,
                                            groups=groups, norm=norm, activation=activation,
                                            use_attention=use_attention,
                                            name=block_name)(branches_out)
    return branches_out


def get_block_5(input_tensor, block_size, padding, groups,
                use_attention, norm, activation, name_prefix):
    branch_0 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(input_tensor)
    branch_1 = Conv2DBN(block_size * 16, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv2DBN(block_size * 16, 3, groups=groups,
                        norm=norm, activation=activation)(branch_1)
    branch_1 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        groups=groups, norm=norm, activation=activation)(branch_1)
    branch_pool = layers.AveragePooling2D(3, strides=2,
                                          padding=padding)(input_tensor)
    branches_output = [branch_0, branch_1, branch_pool]
    branches_output = layers.Concatenate(axis=CHANNEL_AXIS)(branches_output)
    for idx in range(1, 21):
        if idx == 20:
            block_name = f"{name_prefix}block_5"
        else:
            block_name = f'{name_prefix}block_17_{idx}'
        branches_output = InceptionResnetBlock(scale=0.11,
                                               block_type='block17', block_size=block_size,
                                               groups=groups, norm=norm, activation=activation,
                                               use_attention=use_attention,
                                               name=block_name)(branches_output)
    return branches_output


def get_output_block(input_tensor, block_size, padding, groups, use_attention,
                     norm, activation, last_activation, name_prefix):
    branch_0 = Conv2DBN(block_size * 16, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_0 = Conv2DBN(block_size * 24, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(branch_0)
    branch_1 = Conv2DBN(block_size * 16, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_1 = Conv2DBN(block_size * 18, 3, strides=2, padding=padding,
                        norm=norm, groups=groups, activation=activation)(branch_1)
    branch_2 = Conv2DBN(block_size * 16, 1, groups=groups,
                        norm=norm, activation=activation)(input_tensor)
    branch_2 = Conv2DBN(block_size * 18, 3, groups=groups,
                        norm=norm, activation=activation)(branch_2)
    branch_2 = Conv2DBN(block_size * 20, 3, strides=2, padding=padding,
                        groups=groups, norm=norm, activation=activation)(branch_2)
    branch_pool = layers.MaxPooling2D(3, strides=2,
                                      padding=padding)(input_tensor)
    branches_output = [branch_0, branch_1, branch_2, branch_pool]
    branches_output = layers.Concatenate(axis=CHANNEL_AXIS)(branches_output)
    for idx in range(1, 11):
        branches_output = InceptionResnetBlock(scale=0.2,
                                               block_type='block8', block_size=block_size,
                                               groups=groups, norm=norm, activation=activation,
                                               use_attention=use_attention,
                                               name=f'{name_prefix}block_8_{idx}')(branches_output)
    branches_output = Conv2DBN(block_size * 96, 1, groups=groups,
                               norm=norm, activation=last_activation)(branches_output)
    return branches_output


class Conv2DBN(layers.Layer):
    def __init__(self, filters, kernel_size,
                 strides=1, padding="same", groups=1,
                 norm="batch", activation="relu", use_bias=False, name=None):
        super().__init__()
        norm_axis = CHANNEL_AXIS
        self.norm = norm
        if norm == "spectral":
            self.conv_layer = SpectralNormalization(layers.Conv2D(filters=filters,
                                                                  kernel_size=kernel_size,
                                                                  strides=strides,
                                                                  padding=padding,
                                                                  groups=groups,
                                                                  use_bias=use_bias,
                                                                  kernel_initializer='glorot_uniform',
                                                                  bias_initializer='zeros'))
        else:
            if groups == 1:
                # self.conv_layer = EqualizedConv(out_channels=filters,
                #                                 kernel=kernel_size,
                #                                 downsample=strides == 2,
                #                                 padding=padding,
                #                                 use_bias=use_bias)
                self.conv_layer = layers.Conv2D(filters=filters,
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
                self.conv_layer = layers.Conv2D(filters=filters,
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
        out = self.conv_layer(input_tensor)
        if self.norm == "spectral":
            pass
        else:
            out = self.norm_layer(out)
        out = self.act_layer(out)
        return out


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
                 use_attention=True, name=None):
        super().__init__(name=name)
        self.use_attention = use_attention
        if block_type == 'block35':
            branch_0 = Conv2DBN(block_size * 2, 1,
                                groups=groups, norm=norm, activation=activation)
            branch_1_1 = Conv2DBN(block_size * 2, 1,
                                  groups=groups, norm=norm, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 2, 3,
                                  groups=groups, norm=norm, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2])
            branch_2_1 = Conv2DBN(block_size * 2, 1,
                                  groups=groups, norm=norm, activation=activation)
            branch_2_2 = Conv2DBN(block_size * 3, 3,
                                  groups=groups, norm=norm, activation=activation)
            branch_2_3 = Conv2DBN(block_size * 4, 3,
                                  groups=groups, norm=norm, activation=activation)
            branch_2 = Sequential([branch_2_1,
                                   branch_2_2,
                                   branch_2_3])
            branches = [branch_0, branch_1, branch_2]
            up_channel = block_size * 20
        elif block_type == 'block17':
            branch_0 = Conv2DBN(block_size * 12, 1,
                                groups=groups, norm=norm, activation=activation)
            branch_1_1 = Conv2DBN(block_size * 8, 1,
                                  groups=groups, norm=norm, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 10, [1, 7],
                                  groups=groups, norm=norm, activation=activation)
            branch_1_3 = Conv2DBN(block_size * 10, [7, 1],
                                  groups=groups, norm=norm, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3])
            branches = [branch_0, branch_1]
            up_channel = block_size * 68
        elif block_type == 'block8':
            branch_0 = Conv2DBN(block_size * 12, 1,
                                groups=groups, norm=norm, activation=activation)
            branch_1_1 = Conv2DBN(block_size * 12, 1,
                                  groups=groups, norm=norm, activation=activation)
            branch_1_2 = Conv2DBN(block_size * 14, [1, 3],
                                  groups=groups, norm=norm, activation=activation)
            branch_1_3 = Conv2DBN(block_size * 16, [3, 1],
                                  groups=groups, norm=norm, activation=activation)
            branch_1 = Sequential([branch_1_1,
                                   branch_1_2,
                                   branch_1_3])
            branches = [branch_0, branch_1]
            up_channel = block_size * 130
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.branch_layer = ConcatBlock(branches)
        self.up_layer = Conv2DBN(up_channel, 1, groups=groups,
                                 norm=norm, activation=None, use_bias=True)
        if self.use_attention:
            self.attention_layer = CBAM_Block2D(up_channel, ratio=8)
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


class CBAM_Block2D(layers.Layer):
    def __init__(self, input_filter, ratio=8):
        super().__init__()
        self.channel_attention_layer = ChannelAttention2D(input_filter,
                                                          ratio=ratio)
        self.spatial_attention_layer = SpatialAttention2D(input_filter)

    def call(self, input_tensor):
        channel_attention = self.channel_attention_layer(input_tensor)
        spatial_attention = self.spatial_attention_layer(channel_attention)
        return spatial_attention


class ChannelAttention2D(layers.Layer):
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
        self.avg_pool_layer = layers.GlobalAveragePooling2D()
        self.max_pool_layer = layers.GlobalMaxPooling2D()
        self.reshape_layer = layers.Reshape((1, 1, input_filter))
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


class SpatialAttention2D(layers.Layer):
    def __init__(self, input_filter):
        super().__init__()
        kernel_size = 7
        self.avg_pool_layer = layers.Lambda(
            lambda x: backend.mean(x, axis=-1, keepdims=True))
        self.max_pool_layer = layers.Lambda(
            lambda x: backend.max(x, axis=-1, keepdims=True))
        self.concat_layer = layers.Concatenate(axis=-1)
        self.cbam_conv_layer = layers.Conv2D(filters=1,
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


class OutputLayer2D(layers.Layer):
    def __init__(self, last_channel_num, act="tanh"):
        super().__init__()
        USE_CONV_BIAS = True
        self.conv_1x1 = layers.Conv2D(filters=last_channel_num,
                                      kernel_size=1,
                                      padding="same",
                                      strides=1,
                                      use_bias=False,
                                      )
        self.conv_3x3 = layers.Conv2D(filters=last_channel_num,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      use_bias=False,
                                      )
        self.act = get_act_layer(act)

    def call(self, input_tensor):
        conv_1x1 = self.conv_1x1(input_tensor)
        conv_3x3 = self.conv_3x3(input_tensor)
        output = (conv_1x1 + conv_3x3) / 2
        output = self.act(output)

        return output


def DecoderTransposeX2Block(filters):
    return layers.Conv2DTranspose(
        filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        use_bias=USE_CONV_BIAS,
    )


def get_input_label2image_tensor(label_len, target_shape,
                                 dropout_ratio=0.5, reduce_level=5):
    target_channel = target_shape[-1]
    reduce_size = (2 ** reduce_level)
    reduced_shape = (target_shape[0] // reduce_size,
                     target_shape[1] // reduce_size,
                     target_shape[2])
    class_input = layers.Input(shape=(label_len,))
    class_tensor = layers.Dense(np.prod(reduced_shape) // 2,
                                use_bias=USE_DENSE_BIAS)(class_input)
    class_tensor = layers.LeakyReLU(alpha=0.3)(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Dense(np.prod(reduced_shape),
                                use_bias=USE_DENSE_BIAS)(class_tensor)
    class_tensor = layers.LeakyReLU(alpha=0.3)(class_tensor)
    class_tensor = layers.Dropout(dropout_ratio)(class_tensor)
    class_tensor = layers.Reshape(reduced_shape)(class_tensor)
    for idx in range(1, reduce_level):
        class_tensor = DecoderTransposeX2Block(
            min(target_shape[2] * (2 ** idx), 1024))(class_tensor)
        class_tensor = layers.Activation("tanh")(class_tensor)
    class_tensor = DecoderTransposeX2Block(target_channel)(class_tensor)
    class_tensor = layers.Reshape(target_shape)(class_tensor)
    class_tensor = layers.Activation("tanh")(class_tensor)
    return class_input, class_tensor
