import numpy as np
from tensorflow.keras import Model, layers
from . import swin_layers
from . import transformer_layers
from . import utils


def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,
                           act="gelu", shift_window=True, name='swin_unet'):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0  # Droupout after each MLP layer
    attn_drop_rate = 0  # Dropout after Swin-Attention
    # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    proj_drop_rate = 0
    drop_path_rate = 0  # Drop-path within skip-connections

    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

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
                                             name=f'{name}{i}')(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True, name='swin_unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head

    '''
    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x, num_patch_y = utils.get_image_patch_num(input_size[0:2],
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
    X = swin_transformer_stack(X,
                               stack_num=stack_num_down,
                               embed_dim=embed_dim,
                               num_patch=(num_patch_x, num_patch_y),
                               num_heads=num_heads[0],
                               window_size=window_size[0],
                               num_mlp=num_mlp,
                               act=act,
                               shift_window=shift_window,
                               name='{}_swin_down'.format(name))
    print(f"depth {depth} X shape: {X.shape}")
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        print(f"depth {i} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
                                             embed_dim=embed_dim,
                                             name='down{}'.format(i))(X)
        print(f"depth {i} X merging shape: {X.shape}")

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack(X,
                                   stack_num=stack_num_down,
                                   embed_dim=embed_dim,
                                   num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i + 1],
                                   window_size=window_size[i + 1],
                                   num_mlp=num_mlp,
                                   act=act,
                                   shift_window=shift_window,
                                   name='{}_swin_down{}'.format(name, i + 1))

        print(f"depth {i} X Skip shape: {X.shape}")
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
    print(X.shape)
    for i in range(depth_decode):
        print(f"depth decode {i} X shape: {X.shape}")
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                               embed_dim=embed_dim,
                                               upsample_rate=2,
                                               return_vector=True)(X)
        print(f"depth expanding {i} X shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = layers.Dense(embed_dim, use_bias=False,
                         name='{}_concat_linear_proj_{}'.format(name, i))(X)

        # Swin Transformer stacks
        X = swin_transformer_stack(X,
                                   stack_num=stack_num_up,
                                   embed_dim=embed_dim,
                                   num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   num_mlp=num_mlp,
                                   act=act,
                                   shift_window=shift_window,
                                   name='{}_swin_up{}'.format(name, i))
        print(f"depth decode output {i} X shape: {X.shape}")
    print(X.shape)
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    print(X.shape)

    if stride_mode == "half":
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
                                             embed_dim=embed_dim,
                                             name='down_last')(X)
        num_patch_x, num_patch_y = num_patch_x // 2, num_patch_y // 2
        embed_dim *= 2
        X = swin_transformer_stack(X,
                                   stack_num=stack_num_up,
                                   embed_dim=embed_dim,
                                   num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   num_mlp=num_mlp,
                                   act=act,
                                   shift_window=shift_window,
                                   name='{}_swin_down_last'.format(name))
    print(X.shape, num_patch_x, num_patch_y)
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                           embed_dim=embed_dim,
                                           upsample_rate=patch_size[0],
                                           return_vector=False)(X)

    print(X.shape)
    return X


def get_swin_unet_2d(input_shape, last_channel_num,
                     filter_num_begin, depth,
                     stack_num_down, stack_num_up,
                     patch_size, stride_mode, num_heads, window_size, num_mlp,
                     act="gelu", last_act="sigmoid", shift_window=True):
    IN = layers.Input(input_shape)

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                          patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                          shift_window=shift_window, name='swin_unet')
    OUT = layers.Conv2D(last_channel_num, kernel_size=1,
                        use_bias=False, activation=last_act)(X)

    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model


def swin_transformer_stack_3d(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,
                              act="gelu", shift_window=True, name='swin_unet'):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0  # Droupout after each MLP layer
    attn_drop_rate = 0  # Dropout after Swin-Attention
    # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    proj_drop_rate = 0
    drop_path_rate = 0  # Drop-path within skip-connections

    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

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
                                               name=f'{name}{i}')(X)
    return X
