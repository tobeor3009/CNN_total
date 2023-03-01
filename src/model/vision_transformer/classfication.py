import numpy as np
from tensorflow.keras.models import Model
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras import layers
from . import swin_layers, transformer_layers, utils
from .base_layer import swin_transformer_stack_2d, swin_transformer_stack_3d
from .util_layers import DenseLayer
BLOCK_MODE_NAME = "classification"


def swin_classification_2d_base(input_tensor, filter_num_begin, depth, stack_num_per_depth,
                                patch_size, stride_mode, num_heads, window_size, num_mlp,
                                act="gelu", shift_window=True, swin_v2=False, use_sn=False, name="classification"):

    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_size[:-1],
                                                            patch_size,
                                                            stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    # Extract patches from the input tensor
    X = transformer_layers.PatchExtract(patch_size,
                                        stride_size)(input_tensor)

    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
    # -------------------- Swin transformers -------------------- #
    # Stage 1: window-attention + Swin-attention + patch-merging

    for idx in range(depth):
        if idx % 2 == 1:
            shift_window_temp = shift_window
        else:
            shift_window_temp = False

        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_per_depth,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[idx],
                                      window_size=window_size[idx],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window_temp,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_block{}'.format(name, idx))
    # Patch-merging
    #    Pooling patch sequences. Half the number of patches (skip every two patches) and double the embedded dimensions
    X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                        embed_dim=embed_dim,
                                        swin_v2=swin_v2,
                                        use_sn=use_sn,
                                        name='down{}'.format(idx))(X)
    return X


def get_swin_classification_2d(input_shape, last_channel_num,
                               filter_num_begin, depth, stack_num_per_depth,
                               patch_size, stride_mode, num_heads, window_size, num_mlp,
                               act="gelu", last_act="softmax", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    X = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    X = layers.Reshape((h, w, -1))(X)
    X = AdaptiveAveragePooling2D((h // 4, w // 4))(X)
    X = layers.Flatten()(X)
    # The output section
    OUT = DenseLayer(last_channel_num, activation=last_act, use_sn=use_sn)(X)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model


def swin_classification_3d_base(input_tensor, filter_num_begin, depth, stack_num_per_depth,
                                patch_size, stride_mode, num_heads, window_size, num_mlp,
                                act="gelu", shift_window=True, include_3d=False, swin_v2=False, use_sn=False,
                                name="classification"):

    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_z, num_patch_x, num_patch_y = utils.get_image_patch_num_3d(input_size[:-1],
                                                                         patch_size,
                                                                         stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    # Extract patches from the input tensor
    X = transformer_layers.PatchExtract3D(patch_size,
                                          stride_size)(input_tensor)

    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_z * num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
    # -------------------- Swin transformers -------------------- #
    # Stage 1: window-attention + Swin-attention + patch-merging

    for idx in range(depth):

        if idx % 2 == 1:
            shift_window_temp = shift_window
        else:
            shift_window_temp = False
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_per_depth,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_z,
                                                 num_patch_x,
                                                 num_patch_y),
                                      num_heads=num_heads[idx],
                                      window_size=window_size[idx],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window_temp,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_block{}'.format(name, idx))
    # Patch-merging
    #    Pooling patch sequences. Half the number of patches (skip every two patches) and double the embedded dimensions
    X = transformer_layers.PatchMerging3D((num_patch_z, num_patch_x, num_patch_y),
                                          embed_dim=embed_dim,
                                          include_3d=include_3d,
                                          swin_v2=swin_v2,
                                          use_sn=use_sn,
                                          name='down{}'.format(idx))(X)
    return X


def get_swin_classification_3d(input_shape, last_channel_num,
                               filter_num_begin, depth, stack_num_per_depth,
                               patch_size, stride_mode, num_heads, window_size, num_mlp,
                               act="gelu", last_act="softmax", shift_window=True, include_3d=False, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    X = swin_classification_3d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window,
                                    include_3d=include_3d, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    print(f"transformer output shape: {X.shape}")
    X = layers.GlobalAveragePooling1D()(X)
    print(f"GAP shape: {X.shape}")
    # The output section
    OUT = DenseLayer(last_channel_num, activation=last_act, use_sn=use_sn)(X)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model
