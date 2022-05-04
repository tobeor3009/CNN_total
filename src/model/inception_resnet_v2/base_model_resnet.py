import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers_resnet import highway_conv2d, highway_decode2d
from .cbam_attention_module import attach_attention_module
from .base_model import conv2d_bn


def HighWayResnet2D(input_shape=None,
                    block_size=16,
                    groups=1,
                    num_downsample=5,
                    padding="valid",
                    base_act="relu",
                    last_act="relu",
                    name_prefix="",
                    attention_module="cbam_block",
                    **kwargs):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    SKIP_CONNECTION_LAYER_NAMES = [
        f"{name_prefix}conv_down_{idx}_output" for idx in range(num_downsample)]
    input_layer = layers.Input(shape=input_shape)
    x = highway_conv2d(input_tensor=input_layer, filters=block_size * 2,
                       downsample=False, same_channel=False,
                       padding=padding, activation=base_act,
                       groups=groups, name=f"{name_prefix}conv_down_0")
    for idx in range(1, num_downsample + 1):
        filter_size = (block_size * 2) * (2 ** (idx - 1))
        x = highway_conv2d(input_tensor=x, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding=padding, activation=base_act,
                           groups=groups)
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding=padding, activation=base_act, groups=groups)
        if idx == num_downsample - 1:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act, groups=groups)
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, kernel_size=(1, 1),
                               padding=padding, activation=last_act,
                               groups=groups, name=f"{name_prefix}conv_down_{idx}")
        else:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act,
                               groups=groups, name=f"{name_prefix}conv_down_{idx}")

    model = Model(input_layer, x)
    return model, SKIP_CONNECTION_LAYER_NAMES


def HighWayDecoder2D(input_shape=None,
                     block_size=16,
                     num_downsample=5,
                     padding="valid",
                     base_act="relu",
                     last_act="relu",
                     name_prefix="",
                     attention_module="cbam_block",
                     **kwargs):
    if name_prefix == "":
        pass
    else:
        name_prefix = f"{name_prefix}_"
    SKIP_CONNECTION_LAYER_NAMES = [
        f"{name_prefix}conv_down_{idx}_output" for idx in range(num_downsample)]
    input_layer = layers.Input(shape=input_shape)
    x = highway_conv2d(input_tensor=input_layer, filters=block_size * 2,
                       downsample=False, same_channel=False,
                       padding=padding, activation=base_act,
                       name=f"{name_prefix}conv_down_0")
    for idx in range(1, num_downsample + 1):
        filter_size = (block_size * 2) * (2 ** (idx - 1))
        x = highway_conv2d(input_tensor=x, filters=filter_size,
                           downsample=False, same_channel=False,
                           padding=padding, activation=base_act)
        x = highway_conv2d(input_tensor=x, filters=filter_size, downsample=False,
                           padding=padding, activation=base_act)
        if idx == num_downsample - 1:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act)
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, kernel_size=(1, 1),
                               padding=padding, activation=last_act, name=f"{name_prefix}conv_down_{idx}")
        else:
            x = highway_conv2d(input_tensor=x, filters=filter_size * 2, downsample=True,
                               padding=padding, activation=base_act, name=f"{name_prefix}conv_down_{idx}")

    model = Model(input_layer, x)
    return model, SKIP_CONNECTION_LAYER_NAMES
