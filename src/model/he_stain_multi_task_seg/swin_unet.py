import numpy as np
from tensorflow.keras import Model, layers
from ..vision_transformer import swin_layers, transformer_layers, utils
from ..vision_transformer.base_layer import swin_transformer_stack_2d, swin_transformer_stack_3d
from ..inception_resnet_v2_unet_fix.layers import get_act_layer
from .util import BASE_ACT, RGB_OUTPUT_CHANNEL, SEG_OUTPUT_CHANNEL
BLOCK_MODE_NAME = "seg"


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, num_heads, window_size, num_mlp,
                      act="gelu", last_act="sigmoid", image_last_act="tanh", num_class=SEG_OUTPUT_CHANNEL,
                      shift_window=True, swin_v2=False, multi_task=False,
                      name='unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head

    '''
    # Compute number be patches to be embeded
    stride_size = patch_size
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim)(X)
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
    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            name='down{}'.format(i))(X)
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
        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # other tensors are preserved for concatenation
    depth_decode = depth_ - 1
    upsample_x = X
    for i in range(depth_decode):
        # Patch expanding
        upsample_x = transformer_layers.PatchExpandingSimple(num_patch=(num_patch_x, num_patch_y),
                                                             embed_dim=embed_dim,
                                                             upsample_rate=2,
                                                             swin_v2=swin_v2,
                                                             return_vector=True)(upsample_x)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2
        if i < depth_decode - 1:
            # Concatenation and linear projection
            upsample_x = layers.concatenate([upsample_x, X_skip[i]], axis=-1,
                                            name='{}_upsample_concat_{}'.format(name, i))
            upsample_x = layers.Dense(embed_dim, use_bias=False,
                                      name='{}_upsample_concat_linear_proj_{}'.format(name, i))(upsample_x)

        # Swin Transformer stacks
        upsample_x = swin_transformer_stack_2d(upsample_x,
                                               stack_num=stack_num_up,
                                               embed_dim=embed_dim,
                                               num_patch=(
                                                   num_patch_x, num_patch_y),
                                               num_heads=num_heads[i],
                                               window_size=window_size[i],
                                               num_mlp=num_mlp,
                                               act=act,
                                               shift_window=shift_window,
                                               mode=BLOCK_MODE_NAME,
                                               swin_v2=swin_v2,
                                               name='{}_upsample_swin_up{}'.format(name, i))
    upsample_x = transformer_layers.PatchExpandingSimple(num_patch=(num_patch_x, num_patch_y),
                                                         embed_dim=embed_dim,
                                                         upsample_rate=patch_size[0],
                                                         swin_v2=swin_v2,
                                                         return_vector=False)(upsample_x)
    upsample_x = layers.Conv2D(num_class, 1, 1, use_bias=False)(upsample_x)
    upsample_x = get_act_layer(last_act)(upsample_x)
    if multi_task:
        pixel_shuffle_x = X
        for i in range(depth_decode):
            # Patch expanding
            pixel_shuffle_x = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                                                embed_dim=embed_dim,
                                                                upsample_rate=2,
                                                                swin_v2=swin_v2,
                                                                return_vector=True)(pixel_shuffle_x)
            # update token shape info
            embed_dim = embed_dim // 2
            num_patch_x = num_patch_x * 2
            num_patch_y = num_patch_y * 2
            if i < depth_decode - 1:
                # Concatenation and linear projection
                pixel_shuffle_x = layers.concatenate([pixel_shuffle_x, X_skip[i]], axis=-1,
                                                     name='{}_pixel_shuffle_concat_{}'.format(name, i))
                pixel_shuffle_x = layers.Dense(embed_dim, use_bias=False,
                                               name='{}_pixel_shuffle_concat_linear_proj_{}'.format(name, i))(pixel_shuffle_x)

            # Swin Transformer stacks
            pixel_shuffle_x = swin_transformer_stack_2d(pixel_shuffle_x,
                                                        stack_num=stack_num_up,
                                                        embed_dim=embed_dim,
                                                        num_patch=(
                                                            num_patch_x, num_patch_y),
                                                        num_heads=num_heads[i],
                                                        window_size=window_size[i],
                                                        num_mlp=num_mlp,
                                                        act=act,
                                                        shift_window=shift_window,
                                                        mode=BLOCK_MODE_NAME,
                                                        swin_v2=swin_v2,
                                                        name='{}_pixel_shuffle_swin_up{}'.format(name, i))
        pixel_shuffle_x = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                                            embed_dim=embed_dim,
                                                            upsample_rate=patch_size[0],
                                                            swin_v2=swin_v2,
                                                            return_vector=False)(pixel_shuffle_x)
        pixel_shuffle_x = layers.Conv2D(RGB_OUTPUT_CHANNEL, 1, 1,
                                        use_bias=False)(pixel_shuffle_x)
        pixel_shuffle_x = get_act_layer(image_last_act)(pixel_shuffle_x)
    if multi_task:
        return pixel_shuffle_x, upsample_x
    else:
        return upsample_x


def SwinUnet(input_shape=None, class_num=1,
             image_last_act="tanh", last_act="sigmoid", multi_task=False):
    stack_num_down = 2
    stack_num_up = 2
    patch_size = (4, 4)
    # model_parameter ########6
    filter_num_begin = 128
    depth = 3
    num_heads = [4, 8, 8]
    window_size = [4, 2, 2]
    num_mlp = 512
    input_tensor = layers.Input(input_shape)
    unet_output = swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                                    patch_size, num_heads, window_size, num_mlp,
                                    act="gelu", last_act=last_act, image_last_act=image_last_act, num_class=class_num,
                                    shift_window=True, swin_v2=False, multi_task=multi_task, name='unet')
    model = Model(input_tensor, unet_output, name='SwinUnet')
    return model
