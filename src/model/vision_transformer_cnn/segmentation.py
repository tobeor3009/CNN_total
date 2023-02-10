import numpy as np
from tensorflow.keras import Model, layers
from . import swin_layers, transformer_layers, utils
from .base_layer import swin_transformer_stack_2d, swin_transformer_stack_3d

BLOCK_MODE_NAME = "seg"


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, stride_mode, num_heads, window_size, num_mlp,
                      decode_simple=False, act="gelu", shift_window=True, swin_v2=False, name='unet'):
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
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_size[0:2],
                                                            patch_size,
                                                            stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    depth_ = depth

    X_skip = []

    X = input_tensor
    # Patch extraction
    X = transformer_layers.PatchExtract(patch_size,
                                        stride_size)(X)
    print(f"PatchExtract shape: {X.shape}")
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim)(X)
    X = layers.Reshape((num_patch_x, num_patch_y, -1))(X)
    print(f"PatchEmbedding shape: {X.shape}")
    # The first Swin Transformer stack
    X = swin_transformer_stack_2d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(num_patch_x, num_patch_y),
                                  num_heads=num_heads[0],
                                  window_size=window_size[0],
                                  num_mlp=num_mlp,
                                  act=act,
                                  mode=BLOCK_MODE_NAME,
                                  shift_window=shift_window,
                                  swin_v2=swin_v2,
                                  name='{}_swin_down'.format(name))
    print(f"depth {depth} X shape: {X.shape}")
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        print(f"depth {i} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            name='down{}'.format(i))(X)
        print(f"depth {i} X merging shape: {X.shape}")

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[i + 1],
                                      window_size=window_size[i + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
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
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              swin_v2=swin_v2,
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
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[i],
                                      window_size=window_size[i],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      name='{}_swin_up{}'.format(name, i))
        print(f"depth decode output {i} X shape: {X.shape}")
    print(X.shape)
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    print(X.shape)

    if stride_mode == "half":
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            name='down_last')(X)
        num_patch_x, num_patch_y = num_patch_x // 2, num_patch_y // 2
        embed_dim *= 2
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_x, num_patch_y),
                                      num_heads=num_heads[i],
                                      window_size=window_size[i],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      name='{}_swin_down_last'.format(name))
    print(X.shape, num_patch_x, num_patch_y)
    X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            upsample_rate=patch_size[0],
                                            swin_v2=swin_v2,
                                            return_vector=False)(X)

    print(X.shape)
    return X


def get_swin_unet_2d(input_shape, last_channel_num,
                     filter_num_begin, depth,
                     stack_num_down, stack_num_up,
                     patch_size, stride_mode, num_heads, window_size, num_mlp,
                     decode_simple=False, act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False):
    IN = layers.Input(input_shape)

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                          patch_size, stride_mode, num_heads, window_size, num_mlp,
                          decode_simple=decode_simple, act=act, shift_window=shift_window, swin_v2=swin_v2, name='unet')
    OUT = layers.Conv2D(last_channel_num, kernel_size=1,
                        use_bias=False, activation=last_act)(X)

    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model


def swin_unet_3d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, stride_mode, num_heads, window_size, num_mlp,
                      decode_simple=False, act="gelu", shift_window=True, swin_v2=False, name='unet'):
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
    num_patch_z, num_patch_x, num_patch_y = utils.get_image_patch_num_3d(input_size[:-1],
                                                                         patch_size,
                                                                         stride_size)
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    depth_ = depth

    X_skip = []

    X = input_tensor
    # Patch extraction
    X = transformer_layers.PatchExtract3D(patch_size,
                                          stride_size)(X)
    print(f"PatchExtract shape: {X.shape}")
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_z * num_patch_x * num_patch_y,
                                          embed_dim)(X)
    X = layers.Reshape((num_patch_z, num_patch_x, num_patch_y, -1))(X)
    print(f"PatchEmbedding shape: {X.shape}")
    # The first Swin Transformer stack
    X = swin_transformer_stack_3d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(num_patch_z,
                                             num_patch_x,
                                             num_patch_y),
                                  num_heads=num_heads[0],
                                  window_size=window_size[0],
                                  num_mlp=num_mlp,
                                  act=act,
                                  mode=BLOCK_MODE_NAME,
                                  shift_window=shift_window,
                                  swin_v2=swin_v2,
                                  name='{}_swin_down'.format(name))
    print(f"depth {depth} X shape: {X.shape}")
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        print(f"depth {i} X shape: {X.shape}")
        # Patch merging
        X = transformer_layers.PatchMerging3D((num_patch_z, num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              swin_v2=swin_v2,
                                              include_3d=True,
                                              name='down{}'.format(i))(X)
        print(f"depth {i} X merging shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_z = num_patch_z // 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(num_patch_z,
                                                 num_patch_x,
                                                 num_patch_y),
                                      num_heads=num_heads[i + 1],
                                      window_size=window_size[i + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
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
        X = transformer_layers.PatchExpanding3D(num_patch=(num_patch_z,
                                                           num_patch_x,
                                                           num_patch_y),
                                                embed_dim=embed_dim,
                                                upsample_rate=2,
                                                swin_v2=swin_v2,
                                                return_vector=True)(X)
        print(f"depth expanding {i} X shape: {X.shape}")
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_z = num_patch_z * 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
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
                                      swin_v2=swin_v2,
                                      name='{}_swin_up{}'.format(name, i))
        print(f"depth decode output {i} X shape: {X.shape}")
    print(X.shape)
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    print(X.shape)

    if stride_mode == "half":
        X = transformer_layers.PatchMerging3D((num_patch_z, num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              include_3d=True,
                                              name='down_last')(X)
        num_patch_z = num_patch_z // 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2
        embed_dim *= 2
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
                                      name='{}_swin_down_last'.format(name))
    print(X.shape, num_patch_x, num_patch_y)
    X = transformer_layers.PatchExpanding3D(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            upsample_rate=patch_size[0],
                                            swin_v2=swin_v2,
                                            return_vector=False)(X)

    print(X.shape)
    return X


def get_swin_unet_3d(input_shape, last_channel_num,
                     filter_num_begin, depth,
                     stack_num_down, stack_num_up,
                     patch_size, stride_mode, num_heads, window_size, num_mlp,
                     decode_simple=False, act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False):
    IN = layers.Input(input_shape)

    # Base architecture
    X = swin_unet_3d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                          patch_size, stride_mode, num_heads, window_size, num_mlp,
                          decode_simple=decode_simple, act=act, shift_window=shift_window, swin_v2=swin_v2, name='unet')
    OUT = layers.Conv3D(last_channel_num, kernel_size=1,
                        use_bias=False, activation=last_act)(X)

    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model
