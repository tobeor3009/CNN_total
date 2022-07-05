import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers_resnet import highway_conv2d, highway_decode2d, highway_multi
from .cbam_attention_module import attach_attention_module


def HighWayResnet2D(input_shape=None,
                    last_filter=None,
                    block_size=16,
                    groups=1,
                    num_downsample=5,
                    base_act="relu",
                    last_act="relu",
                    name_prefix="",
                    attention_module="cbam_block",
                    **kwargs):
    padding = "same"
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    SKIP_CONNECTION_LAYER_NAMES = [f"{name_prefix}conv_down_{idx}_output"
                                   for idx in range(num_downsample)]
    input_layer = layers.Input(shape=input_shape)
    if groups == 1:
        init_filter = block_size * 2
    else:
        init_filter = block_size * groups

    x = highway_conv2d(input_tensor=input_layer, filters=init_filter,
                       downsample=False, same_channel=False,
                       padding=padding, activation=base_act,
                       groups=groups)
    groups = 1
    for idx in range(0, num_downsample):
        filter_size = (block_size * 2) * (2 ** idx)
        x = highway_conv2d(input_tensor=x, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding=padding, activation=base_act,
                           groups=groups, name=f"{name_prefix}conv_down_{idx}_0")
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding=padding, activation=base_act, groups=groups,
                           name=f"{name_prefix}conv_down_same_{idx}_1")
        if idx == num_downsample - 1:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act, groups=groups,
                               name=f"{name_prefix}conv_down_same_{idx}_2")
            if last_filter is None:
                last_filter = filter_size * 2
                last_same_channel = True
            else:
                last_same_channel = False
            x = highway_conv2d(input_tensor=x, filters=last_filter, kernel_size=(1, 1),
                               same_channel=last_same_channel, padding=padding, activation=last_act,
                               groups=groups, name=f"{name_prefix}conv_down_{idx}")
        else:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act,
                               groups=groups, name=f"{name_prefix}conv_down_{idx}")

    model = Model(input_layer, x, name="encoder")
    return model, SKIP_CONNECTION_LAYER_NAMES


def HighWayDecoder2D(input_tensor=None,
                     encoder=None,
                     skip_connection_layer_names=None,
                     last_filter=None,
                     block_size=16,
                     num_downsample=5,
                     base_act="relu",
                     last_act="relu",
                     name_prefix="",
                     **kwargs):
    padding = "same"
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    x = highway_conv2d(input_tensor=input_tensor, filters=block_size * 2,
                       downsample=False, same_channel=False,
                       padding=padding, activation=base_act)
    for idx in range(num_downsample - 1, -1, -1):
        filter_size = (block_size * 2) * (2 ** idx)
        if skip_connection_layer_names is not None:
            skip_connect = encoder.get_layer(
                skip_connection_layer_names[idx]).output
            x = layers.Concatenate(axis=-1)([x, skip_connect])
        x = highway_conv2d(input_tensor=x, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding=padding, activation=base_act,
                           name=f"{name_prefix}conv_up_same_{idx}_0")
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding=padding, activation=base_act,
                           name=f"{name_prefix}conv_up_same_{idx}_1")
        if idx == 0:
            x = highway_decode2d(input_tensor=x, filters=filter_size * 2, unsharp=True,
                                 activation=base_act,
                                 name=f"{name_prefix}conv_up_same_{idx}_2")
            if last_filter is None:
                last_filter = filter_size * 2
                last_same_channel = True
            else:
                last_same_channel = False
            x = highway_conv2d(input_tensor=x, filters=last_filter, kernel_size=(1, 1), same_channel=last_same_channel,
                               padding=padding, activation=last_act, name=f"{name_prefix}conv_up_{idx}")
        else:
            x = highway_decode2d(input_tensor=x, filters=filter_size * 2, unsharp=True,
                                 activation=base_act, name=f"{name_prefix}conv_up_{idx}")
    return x


def HighWayResnet2D_Progressive(input_shape=None,
                                last_filter=None,
                                block_size=16,
                                groups=1,
                                num_downsample=5,
                                final_downsample=5,
                                base_act="relu",
                                last_act="relu",
                                name_prefix="",
                                attention_module="cbam_block",
                                **kwargs):
    padding = "same"
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    SKIP_CONNECTION_LAYER_NAMES = [f"{name_prefix}conv_down_{final_downsample - idx - 1}_output"
                                   for idx in range(final_downsample - num_downsample, final_downsample)]
    input_layer = layers.Input(shape=input_shape)

    x = input_layer
    for idx in range(final_downsample - num_downsample, final_downsample):
        filter_size = (block_size * 2) * (2 ** idx)
        layer_idx = final_downsample - idx - 1
        x = highway_conv2d(input_tensor=x, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding=padding, activation=base_act,
                           groups=groups, name=f"{name_prefix}conv_down_{layer_idx}_0")
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding=padding, activation=base_act, groups=groups,
                           name=f"{name_prefix}conv_down_same_{layer_idx}_1")
        if idx == num_downsample - 1:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act, groups=groups,
                               name=f"{name_prefix}conv_down_same_{layer_idx}_2")
            if last_filter is None:
                last_filter = filter_size * 2
                last_same_channel = True
            else:
                last_same_channel = False
            x = highway_conv2d(input_tensor=x, filters=last_filter, kernel_size=(1, 1),
                               same_channel=last_same_channel, padding=padding, activation=last_act,
                               groups=groups, name=f"{name_prefix}conv_down_{layer_idx}")
        else:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act,
                               groups=groups, name=f"{name_prefix}conv_down_{layer_idx}")

    model = Model(input_layer, x, name="encoder")
    return model, SKIP_CONNECTION_LAYER_NAMES
