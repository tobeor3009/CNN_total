import numpy as np
from tensorflow.keras import Model, layers
from . import swin_layers, transformer_layers, utils


def swin_transformer_stack_2d(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,
                              act, shift_window, mode, swin_v2=False, name=''):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''

    if mode == "seg":
        # Turn-off dropouts
        mlp_drop_rate = 0  # Droupout after each MLP layer
        attn_drop_rate = 0  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0
        drop_path_rate = 0  # Drop-path within skip-connections
    elif mode == "classification":
        # Dropout parameters
        mlp_drop_rate = 0.01  # Droupout after each MLP layer
        attn_drop_rate = 0.01  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0.01
        drop_path_rate = 0.01  # Drop-path within skip-connections
    else:
        assert False, "block mode must in ('seg', 'classification')"
    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

    if isinstance(window_size, int):
        window_size = np.array((window_size, window_size))
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0

    for i in range(stack_num):

        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim,
                                             num_patch=num_patch,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             shift_size=shift_size_temp,
                                             num_mlp=num_mlp,
                                             act=act,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate,
                                             attn_drop=attn_drop_rate,
                                             proj_drop=proj_drop_rate,
                                             drop_path_prob=drop_path_rate,
                                             swin_v2=swin_v2,
                                             name=f'{name}{i}')(X)
    return X


def swin_context_transformer_stack_2d(X, Y, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,
                                      act, shift_window, mode, swin_v2=False, name=''):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''

    if mode == "seg":
        # Turn-off dropouts
        mlp_drop_rate = 0  # Droupout after each MLP layer
        attn_drop_rate = 0  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0
        drop_path_rate = 0  # Drop-path within skip-connections
    elif mode == "classification":
        # Dropout parameters
        mlp_drop_rate = 0.01  # Droupout after each MLP layer
        attn_drop_rate = 0.01  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0.01
        drop_path_rate = 0.01  # Drop-path within skip-connections
    else:
        assert False, "block mode must in ('seg', 'classification')"
    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

    if isinstance(window_size, int):
        window_size = np.array((window_size, window_size))
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0

    for i in range(stack_num):

        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size
        X = swin_layers.ContextSwinTransformerBlock(dim=embed_dim,
                                                    num_patch=num_patch,
                                                    num_heads=num_heads,
                                                    window_size=window_size,
                                                    shift_size=shift_size_temp,
                                                    num_mlp=num_mlp,
                                                    act=act,
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    mlp_drop=mlp_drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    proj_drop=proj_drop_rate,
                                                    drop_path_prob=drop_path_rate,
                                                    swin_v2=swin_v2,
                                                    name=f'{name}{i}')(X, Y)
    return X


def swin_transformer_stack_3d(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,
                              act, shift_window, mode, swin_v2=False, name=''):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''

    if mode == "seg":
        # Turn-off dropouts
        mlp_drop_rate = 0  # Droupout after each MLP layer
        attn_drop_rate = 0  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0
        drop_path_rate = 0  # Drop-path within skip-connections
    elif mode == "classification":
        # Dropout parameters
        mlp_drop_rate = 0.01  # Droupout after each MLP layer
        attn_drop_rate = 0.01  # Dropout after Swin-Attention
        # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        proj_drop_rate = 0.01
        drop_path_rate = 0.01  # Drop-path within skip-connections
    else:
        assert False, "block mode must in ('seg', 'classification')"
    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

    if isinstance(window_size, int):
        window_size = np.array((window_size, window_size, window_size))
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0

    for i in range(stack_num):

        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock3D(dim=embed_dim,
                                               num_patch=num_patch,
                                               num_heads=num_heads,
                                               window_size=window_size,
                                               shift_size=shift_size_temp,
                                               num_mlp=num_mlp,
                                               act=act,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               mlp_drop=mlp_drop_rate,
                                               attn_drop=attn_drop_rate,
                                               proj_drop=proj_drop_rate,
                                               drop_path_prob=drop_path_rate,
                                               swin_v2=swin_v2,
                                               name=f'{name}{i}')(X)
    return X
