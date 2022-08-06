import numpy as np
import math

from tensorflow.keras import layers, Model

from .base_layer import swin_transformer_stack_2d, swin_transformer_stack_3d
from . import utils, transformer_layers

BLOCK_MODE_NAME = "seg"


def skip_connect_expanding(decoded, skip_connect):
    decode_embed_dim = decoded.shape[2]
    skip_patch_num = skip_connect.shape[1]
    embed_dim = decode_embed_dim

    num_skip_patch_x = math.sqrt(skip_patch_num)
    num_skip_patch_y = math.sqrt(skip_patch_num)
    assert int(
        num_skip_patch_x) == num_skip_patch_x, f"num_skip_patch_x: {num_skip_patch_x}"
    num_skip_patch_x = int(num_skip_patch_x)
    num_skip_patch_y = int(num_skip_patch_y)

    skip_connect = transformer_layers.patch_expanding_2d_3d(num_patch=(num_skip_patch_x, num_skip_patch_y),
                                                            embed_dim=embed_dim,
                                                            return_vector=True)(skip_connect)

    skip_connect = layers.Dense(decode_embed_dim, use_bias=False)(skip_connect)
    return skip_connect


def swin_x2ct_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                   patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu",
                   shift_window=True, name="swin_x2ct"):

    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_size[0:2],
                                                            patch_size,
                                                            stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    depth_ = depth

    X_skip = []

    X = input_tensor
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size,
                                         stride_size)(X)
    print(f"patch_extract shape: {X.shape}")
    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x * num_patch_y,
                                           embed_dim)(X)
    print(f"patch_embedding shape: {X.shape}")
    # The first Swin Transformer stack
    X = swin_transformer_stack_2d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(num_patch_x, num_patch_y),
                                  num_heads=num_heads[0],
                                  window_size=window_size[0],
                                  num_mlp=num_mlp,
                                  act=act,
                                  shift_window=shift_window,
                                  mode=BLOCK_MODE_NAME,
                                  name='{}_swin_down'.format(name))
    print(f"depth {depth} X shape: {X.shape}")
    X_skip.append(X)

    # Downsampling blocks
    for idx in range(depth_ - 1):
        print(f"depth {idx} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
                                             embed_dim=embed_dim,
                                             name='down{}'.format(idx))(X)
        print(f"depth {idx} X merging shape: {X.shape}")

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[idx + 1],
                                      window_size=window_size[idx + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      name='{}_swin_down{}'.format(name, idx + 1))

        print(f"depth {idx} X Skip shape: {X.shape}")
        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    print(f"dedoced shape: {X.shape}")

    expand_ratio = math.log(num_patch_x, 4)
    assert int(expand_ratio) == expand_ratio, f"expand_ratio: {expand_ratio}"
    need_ct_upsacle = int(expand_ratio)

    for _ in range(need_ct_upsacle):
        print(f"depth middle {idx} X shape: {X.shape}")
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                               embed_dim=embed_dim,
                                               upsample_rate=2,
                                               return_vector=True)(X)
        print(f"depth middle expanding {idx} X shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        X = layers.Dense(embed_dim, use_bias=False)(X)

        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[idx],
                                      window_size=window_size[idx],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      name='')

    num_patch_x = num_patch_x // (2 ** need_ct_upsacle)
    num_patch_y = num_patch_y // (2 ** need_ct_upsacle)
    num_patch_z = num_patch_x
    print(f"ct decoded shape: {X.shape}")

    for i in range(depth_decode):
        print(f"depth {i} decode X shape: {X.shape}")
        # Patch expanding
        X = transformer_layers.patch_expanding_3d(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                                  embed_dim=embed_dim,
                                                  upsample_rate=2,
                                                  return_vector=True)(X)
        skip_connect = skip_connect_expanding(X, X_decode[i])

        print(f"depth {i} expanding X shape: {X.shape}")
        print(f"depth {i} skip_connect X shape: {skip_connect.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2
        num_patch_z = num_patch_z * 2
        # Concatenation and linear projection
        X = layers.concatenate([X, skip_connect], axis=-1,
                               name='{}_concat_{}'.format(name, i))
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
                                      name='{}_swin_up{}'.format(name, i))
        print(f"depth {i} decode output X shape: {X.shape}")

    X = transformer_layers.patch_expanding_3d(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=patch_size[0],
                                              return_vector=False)(X)
    return X


def get_swin_x2ct(input_shape, last_channel_num,
                  filter_num_begin, depth,
                  stack_num_down, stack_num_up,
                  patch_size, stride_mode, num_heads, window_size, num_mlp,
                  act="gelu", last_act="sigmoid", shift_window=True):
    IN = layers.Input(input_shape)

    X = swin_x2ct_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                       patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                       shift_window=shift_window, name="swin_x2ct")
    OUT = layers.Conv2D(last_channel_num, kernel_size=1,
                        use_bias=False, activation=last_act)(X)
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model
