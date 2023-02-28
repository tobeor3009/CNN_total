import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import AdaptiveAveragePooling1D, AdaptiveAveragePooling2D
from . import swin_layers, transformer_layers, utils
from .base_layer import swin_transformer_stack_2d, swin_context_transformer_stack_2d
from .classfication import swin_classification_2d_base, swin_classification_3d_base, get_swin_classification_2d
from ..inception_resnet_v2_unet_fix.base_model_as_class import InceptionResNetV2_progressive
from .util_layers import DenseLayer, Conv1DLayer, Conv2DLayer
BLOCK_MODE_NAME = "seg"


# Get and Revise From https://github.com/LynnHo/AttGAN-Tensorflow/blob/master/utils.py
def tile_concat_1d(a_list, b_list=[]):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        class_num = b.shape[-1]
        # padding_num = (4 - class_num) % 4
        # if padding_num != 0:
        #     paddings = [[0, 0], [padding_num, 0]]
        #     b = tf.pad(b, paddings, "CONSTANT")
        #     class_num += padding_num
        b = tf.reshape(b, [-1, 1, class_num])
        b = tf.tile(b, [1, a_list[0].shape[1], 1])
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)


def tile_concat_2d(a_list, b_list=[]):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        class_num = b.shape[-1]
        b = tf.reshape(b, [-1, 1, 1, class_num])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)


def swin_class_gen_2d_base_v1(input_tensor, class_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                              patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                              swin_v2=False, use_sn=False, name='unet'):
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
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
                                  use_sn=use_sn,
                                  name='{}_swin_down'.format(name))
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_down{}'.format(name, i + 1))

        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # upsampling begins at the deepest available tensor
    X = tile_concat_1d(X_skip[0], class_tensor)
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              return_vector=True,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name=f'{name}_swin_expanding_{i}')(X)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = DenseLayer(embed_dim, use_bias=False, use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_up{}'.format(name, i))
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    if stride_mode == "half":
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim, use_sn=use_sn,
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
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_down_last'.format(name))
    X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                          embed_dim=embed_dim,
                                          upsample_rate=patch_size[0],
                                          return_vector=False,
                                          swin_v2=swin_v2,
                                          name=f'{name}_swin_expanding_last')(X)
    return X


def swin_class_gen_2d_base_v2(input_tensor, class_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                              patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                              swin_v2=False, use_sn=False, name='unet'):
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
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
                                  use_sn=use_sn,
                                  name='{}_swin_down'.format(name))
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_down{}'.format(name, i + 1))
        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    # upsampling begins at the deepest available tensor
    X = swin_context_transformer_stack_2d(X, class_tensor,
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
                                          use_sn=use_sn,
                                          name=f'{name}_swin_context')
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              return_vector=True,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name=f'{name}_swin_expanding_{i}')(X)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = DenseLayer(embed_dim, use_bias=False,
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
                                      use_sn=use_sn,
                                      name='{}_swin_up{}'.format(name, i))
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
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
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_down_last'.format(name))
    X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                          embed_dim=embed_dim,
                                          upsample_rate=patch_size[0],
                                          return_vector=False,
                                          swin_v2=swin_v2,
                                          use_sn=use_sn,
                                          name=f'{name}_swin_expanding_last')(X)
    return X


def swin_class_gen_2d_base_v3(input_tensor, out_class_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                              patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                              swin_v2=False, use_sn=False, name='unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head

    '''
    num_class = out_class_tensor.shape[-1]
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
    X = tile_concat_1d(X[..., :-num_class], out_class_tensor)
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
                                  use_sn=use_sn,
                                  name='{}_swin_down'.format(name))
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            use_sn=use_sn,
                                            name='down{}'.format(i))(X)
        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2
        # Swin Transformer stacks
        X = tile_concat_1d(X[..., :-num_class], out_class_tensor)
        X = swin_transformer_stack_2d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(
                                          num_patch_x, num_patch_y),
                                      num_heads=num_heads[i + 1],
                                      window_size=window_size[i + 1],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_down{}'.format(name, i + 1))

        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    X = tile_concat_1d(X[..., :-num_class], out_class_tensor)
    # upsampling begins at the deepest available tensor
    X = swin_transformer_stack_2d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(
                                      num_patch_x, num_patch_y),
                                  num_heads=num_heads[i + 1],
                                  window_size=window_size[i + 1],
                                  num_mlp=num_mlp,
                                  act=act,
                                  shift_window=shift_window,
                                  mode=BLOCK_MODE_NAME,
                                  swin_v2=swin_v2,
                                  use_sn=use_sn,
                                  name='{}_swin_down{}'.format(name, i + 1))
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              return_vector=True,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name=f'{name}_swin_expanding_{i}')(X)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = DenseLayer(embed_dim, use_bias=False, use_sn=use_sn,
                       name='{}_concat_linear_proj_{}'.format(name, i))(X)
        # Swin Transformer stacks
        X = swin_transformer_stack_2d(X,
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
                                      use_sn=use_sn,
                                      name='{}_swin_up{}'.format(name, i))
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    if stride_mode == "half":
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim, use_sn=use_sn,
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
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_down_last'.format(name))
    X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                          embed_dim=embed_dim,
                                          upsample_rate=patch_size[0],
                                          return_vector=False,
                                          swin_v2=swin_v2,
                                          use_sn=use_sn,
                                          name=f'{name}_swin_expanding_last')(X)
    return X


def swin_style_extractor_base(input_tensor, class_tensor, filter_num_begin, depth, stack_num_down,
                              patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                              swin_v2=False, use_sn=False, name='unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head

    '''
    num_class = class_tensor.shape[-1]
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
    X = input_tensor
    # Patch extraction
    X = transformer_layers.PatchExtract(patch_size,
                                        stride_size)(X)
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
    Y = tile_concat_1d(X[..., :-num_class], class_tensor)
    # The first Swin Transformer stack
    X = swin_context_transformer_stack_2d(X, Y,
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
                                          use_sn=use_sn,
                                          name='{}_swin_down'.format(name))
    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            use_sn=use_sn,
                                            name='down{}'.format(i))(X)
        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x = num_patch_x // 2
        num_patch_y = num_patch_y // 2

        # Swin Transformer stacks
        Y = tile_concat_1d(X[..., :-num_class], class_tensor)
        X = swin_context_transformer_stack_2d(X, Y,
                                              stack_num=stack_num_down,
                                              embed_dim=embed_dim,
                                              num_patch=(
                                                  num_patch_x, num_patch_y),
                                              num_heads=num_heads[i + 1],
                                              window_size=window_size[i + 1],
                                              num_mlp=num_mlp,
                                              act=act,
                                              shift_window=shift_window,
                                              mode=BLOCK_MODE_NAME,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name='{}_swin_down{}'.format(name, i + 1))
    return X


def swin_seg_disc_v1(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                     patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                     swin_v2=False, use_sn=False, name='unet'):
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
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
                                  use_sn=use_sn,
                                  name='{}_swin_down'.format(name))
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim,
                                            swin_v2=swin_v2,
                                            use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_down{}'.format(name, i + 1))

        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # upsampling begins at the deepest available tensor
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
                                  use_sn=use_sn,
                                  name='{}_swin_down{}'.format(name, i + 1))
    CLASS_FEATURE = X
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              upsample_rate=2,
                                              return_vector=True,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name=f'{name}_swin_expanding_{i}')(X)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_x = num_patch_x * 2
        num_patch_y = num_patch_y * 2

        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = DenseLayer(embed_dim, use_bias=False, use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_up{}'.format(name, i))
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    if stride_mode == "half":
        X = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                            embed_dim=embed_dim, use_sn=use_sn,
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
                                      use_sn=use_sn,
                                      name='{}_swin_down_last'.format(name))
    X = tf.reshape(X, (-1, input_size[0] // patch_size[0],
                       input_size[1] // patch_size[1], embed_dim))
    return X, CLASS_FEATURE


def get_inception_resnet_v2_class_inputdisc_2d(input_shape,
                                               class_num,
                                               block_size=16,
                                               num_downsample=5,
                                               norm="instance",
                                               padding="valid",
                                               validity_act=None,
                                               base_act="leakyrelu",
                                               last_act="sigmoid",
                                               ):

    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2] + class_num)
    image_input = layers.Input(input_shape)
    class_input = layers.Input(class_num)
    model_input = tile_concat_2d(image_input, class_input)
    base_model = InceptionResNetV2_progressive(target_shape=target_shape,
                                               block_size=block_size,
                                               padding=padding,
                                               norm=norm,
                                               base_act=base_act,
                                               last_act=last_act,
                                               name_prefix="validity",
                                               num_downsample=num_downsample,
                                               use_attention=True)

    model_output = base_model(model_input)

    validity_pred = layers.Conv2D(1, kernel_size=1,
                                  activation=validity_act, kernel_initializer="zeros", use_bias=False)(model_output)
    validity_pred = AdaptiveAveragePooling2D((8, 8))(validity_pred)
    model = Model([image_input, class_input], validity_pred)

    return model


def get_inception_resnet_v2_disc_2d(input_shape,
                                    block_size=16,
                                    num_downsample=5,
                                    norm="instance",
                                    padding="valid",
                                    validity_act=None,
                                    base_act="leakyrelu",
                                    last_act="sigmoid",
                                    ):

    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2])
    image_input = layers.Input(input_shape)
    base_model = InceptionResNetV2_progressive(target_shape=target_shape,
                                               block_size=block_size,
                                               padding=padding,
                                               norm=norm,
                                               base_act=base_act,
                                               last_act=last_act,
                                               name_prefix="validity",
                                               num_downsample=num_downsample,
                                               use_attention=True)

    model_output = base_model(image_input)

    validity_pred = layers.Conv2D(1, kernel_size=1,
                                  activation=validity_act, kernel_initializer="zeros", use_bias=False)(model_output)
    validity_pred = AdaptiveAveragePooling2D((8, 8))(validity_pred)
    model = Model(image_input, validity_pred)

    return model


def get_swin_disc_2d_2d(input_shape, last_channel_num,
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
    X = Conv2DLayer(filter_num_begin, kernel_size=3,
                    activation=act, use_sn=use_sn, padding="same")(X)
    # X = AdaptiveAveragePooling2D((h // 4, w // 4))(X)
    X = AdaptiveAveragePooling2D((8, 8))(X)
    # The output section
    OUT = Conv2DLayer(last_channel_num, kernel_size=1,
                      activation=last_act, use_sn=use_sn,
                      kernel_initializer="zeros", use_bias=False)(X)
    # Model configuration
    model = Model(inputs=IN, outputs=[OUT, ])
    return model


def get_swin_disc_2d_1d(input_shape, class_num, last_channel_num,
                        filter_num_begin, depth, stack_num_per_depth,
                        patch_size, stride_mode, num_heads, window_size, num_mlp,
                        act="gelu", last_act="softmax", shift_window=True, swin_v2=False, use_sn=use_sn):
    H, W, _ = input_shape
    IN = layers.Input(input_shape)
    CLASS = layers.Input((class_num,))
    # Base architecture
    VALIDITY = swin_style_extractor_base(IN, CLASS, filter_num_begin, depth, stack_num_per_depth,
                                         patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                         shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')
    VALIDITY = Conv1DLayer(last_channel_num, kernel_size=1,
                           activation=last_act, use_sn=use_sn,
                           kernel_initializer="zeros", use_bias=False)(VALIDITY)
    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[VALIDITY, ])
    return model


def get_swin_disc_2d_1d_v1(input_shape, class_num, last_channel_num,
                           filter_num_begin, depth, stack_num_per_depth,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    CLASS = layers.Input((class_num,))
    # Base architecture
    VALIDITY = swin_style_extractor_base(IN, CLASS, filter_num_begin, depth, stack_num_per_depth,
                                         patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                         shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')
    VALIDITY = Conv1DLayer(last_channel_num, kernel_size=1,
                           activation=last_act, use_sn=use_sn,
                           kernel_initializer="zeros", use_bias=False)(VALIDITY)
    LABEL = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                        patch_size, stride_mode, num_heads, window_size, num_mlp,
                                        act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                        name="classification")
    LABEL = AdaptiveAveragePooling1D((h * w // 16))(LABEL)
    LABEL = DenseLayer(filter_num_begin, activation=act, use_sn=use_sn)(LABEL)
    LABEL = layers.Flatten()(LABEL)
    LABEL = DenseLayer(class_num, activation='sigmoid', use_sn=use_sn)(LABEL)
    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[VALIDITY, LABEL])
    return model


def get_swin_disc_2d_1d_v2(input_shape, class_num, last_channel_num,
                           filter_num_begin, depth, stack_num_per_depth,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    # Base architecture
    VALIDITY = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                                           act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                           name="classification")
    VALIDITY = AdaptiveAveragePooling1D((h * w // 16))(VALIDITY)
    VALIDITY = DenseLayer(
        filter_num_begin, activation=last_act, use_sn=use_sn)(VALIDITY)

    LABEL = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                        patch_size, stride_mode, num_heads, window_size, num_mlp,
                                        act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name="classification")
    LABEL = AdaptiveAveragePooling1D((h * w // 16))(LABEL)
    LABEL = DenseLayer(filter_num_begin, activation=act, use_sn=use_sn)(LABEL)
    LABEL = layers.Flatten()(LABEL)
    LABEL = DenseLayer(class_num, activation='sigmoid', use_sn=use_sn)(LABEL)
    # Model configuration
    model = Model(inputs=[IN], outputs=[VALIDITY, LABEL])
    return model


def get_seg_swin_disc_2d_v1(input_shape, class_num,
                            filter_num_begin, depth, stack_num_down, stack_num_up,
                            patch_size, stride_mode, num_heads, window_size, num_mlp,
                            act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    # Base architecture
    VALIDITY, LABEL = swin_seg_disc_v1(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                                       patch_size, stride_mode, num_heads, window_size, num_mlp,
                                       act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name="unet")
    VALIDITY = AdaptiveAveragePooling2D((8, 8))(VALIDITY)
    VALIDITY = DenseLayer(
        filter_num_begin, activation=last_act, use_sn=use_sn)(VALIDITY)

    LABEL = AdaptiveAveragePooling1D((h * w // 16))(LABEL)
    LABEL = DenseLayer(filter_num_begin, activation=act, use_sn=use_sn)(LABEL)
    LABEL = layers.Flatten()(LABEL)
    LABEL = DenseLayer(class_num, activation='sigmoid', use_sn=use_sn)(LABEL)
    # Model configuration
    model = Model(inputs=[IN], outputs=[VALIDITY, LABEL])
    return model


def get_seg_swin_disc_2d_v2(input_shape,
                            filter_num_begin, depth, stack_num_down, stack_num_up,
                            patch_size, stride_mode, num_heads, window_size, num_mlp,
                            act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    # Base architecture
    VALIDITY, _ = swin_seg_disc_v1(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                                   patch_size, stride_mode, num_heads, window_size, num_mlp,
                                   act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name="unet")
    VALIDITY = AdaptiveAveragePooling2D((8, 8))(VALIDITY)
    VALIDITY = DenseLayer(
        filter_num_begin, activation=last_act, use_sn=use_sn)(VALIDITY)

    model = Model(inputs=[IN], outputs=[VALIDITY])
    return model


class SwinDiscriminator(Model):
    def __init__(
        self, input_shape, classifier,
        filter_num_begin, depth, stack_num_down, stack_num_up,
        patch_size, stride_mode, num_heads, window_size, num_mlp,
        act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False
    ):
        super().__init__()
        H, W, _ = input_shape
        h, w = H // (2 ** depth), W // (2 ** depth)
        VALIDITY_IN = layers.Input(input_shape)
        VALIDITY, _ = swin_seg_disc_v1(VALIDITY_IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                                       patch_size, stride_mode, num_heads, window_size, num_mlp,
                                       act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name="unet")
        VALIDITY = AdaptiveAveragePooling2D((8, 8))(VALIDITY)
        VALIDITY = DenseLayer(filter_num_begin,
                              activation=last_act, use_sn=use_sn)(VALIDITY)
        self.validity_model = Model(inputs=[VALIDITY_IN], outputs=[VALIDITY])
        # Model configuration
        self.classifier_model = classifier

    def build(self, input_shape):
        super().build(input_shape)
        self.validity_model.build(input_shape)
        self.classifier_model.build(input_shape)

    def call(self, input_tensor):
        validity = self.validity_model(input_tensor)
        label = self.classifier_model(input_tensor)
        return validity, label


def get_swin_class_input_disc_2d(input_shape, class_num, last_channel_num,
                                 filter_num_begin, depth, stack_num_per_depth,
                                 patch_size, stride_mode, num_heads, window_size, num_mlp,
                                 act="gelu", last_act="softmax", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    CLASS = layers.Input(class_num)
    X = tile_concat_2d(IN, CLASS)
    X = swin_classification_2d_base(X, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    X = layers.Reshape((h, w, -1))(X)
    # X = AdaptiveAveragePooling2D((h // 4, w // 4))(X)
    X = AdaptiveAveragePooling2D((8, 8))(X)
    # The output section
    OUT = Conv2DLayer(last_channel_num, kernel_size=1,
                      activation=last_act, use_sn=use_sn,
                      kernel_initializer="zeros", use_bias=False)(X)
    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[OUT, ])
    return model


def get_swin_class_gen_2d_v1(input_shape, class_num, last_channel_num,
                             filter_num_begin, depth,
                             stack_num_down, stack_num_up,
                             patch_size, stride_mode, num_heads, window_size, num_mlp,
                             act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    CLASS = layers.Input(class_num)
    # Base architecture
    X = swin_class_gen_2d_base_v1(IN, CLASS, filter_num_begin, depth, stack_num_down, stack_num_up,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                  shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')

    OUT = Conv2DLayer(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act, use_sn=use_sn)(X)

    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[OUT, ])
    return model


def get_swin_class_gen_2d_v2(input_shape, last_channel_num,
                             filter_num_begin, depth,
                             stack_num_down, stack_num_up,
                             patch_size, stride_mode, num_heads, window_size, num_mlp,
                             act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_shape[0:2],
                                                            patch_size,
                                                            stride_size)
    num_patch_x = num_patch_x // (2 ** (depth - 1))
    num_patch_y = num_patch_y // (2 ** (depth - 1))
    CLASS = layers.Input((num_patch_x * num_patch_y,
                          filter_num_begin * (2 ** (depth - 1))))
    # Base architecture
    X = swin_class_gen_2d_base_v2(IN, CLASS, filter_num_begin, depth, stack_num_down, stack_num_up,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                  shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')
    OUT = Conv2DLayer(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act, use_sn=use_sn)(X)

    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[OUT, ])
    return model


def get_swin_class_gen_2d_v3(input_shape, class_num, last_channel_num,
                             filter_num_begin, depth,
                             stack_num_down, stack_num_up,
                             patch_size, stride_mode, num_heads, window_size, num_mlp,
                             act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_shape[0:2],
                                                            patch_size,
                                                            stride_size)
    num_patch_x = num_patch_x // (2 ** (depth - 1))
    num_patch_y = num_patch_y // (2 ** (depth - 1))
    OUT_CLASS = layers.Input(class_num)
    # Base architecture
    X = swin_class_gen_2d_base_v3(IN, OUT_CLASS, filter_num_begin, depth, stack_num_down, stack_num_up,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                  shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')
    OUT = Conv2DLayer(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act, use_sn=use_sn)(X)

    # Model configuration
    model = Model(inputs=[IN, OUT_CLASS], outputs=[OUT, ])
    return model


def get_swin_style_encoder(input_shape, class_num,
                           filter_num_begin, depth, stack_num_down,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", shift_window=True, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    CLASS = layers.Input((class_num,))
    # Base architecture
    STYLE_LATENT = swin_style_extractor_base(IN, CLASS, filter_num_begin, depth, stack_num_down,
                                             patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                             shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet')
    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[STYLE_LATENT, ])
    return model


def get_swin_class_disc_2d(input_shape, last_channel_num,
                           filter_num_begin, depth, stack_num_per_depth,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    H, W, _ = input_shape
    h, w = H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    X = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    VALIDITY = layers.Reshape((h, w, -1))(X)
    VALIDITY = Conv2DLayer(filter_num_begin, kernel_size=3,
                           activation=act, padding="same", use_sn=use_sn)(VALIDITY)
    VALIDITY = AdaptiveAveragePooling2D((8, 8))(VALIDITY)
    VALIDITY = Conv2DLayer(last_channel_num, kernel_size=1,
                           activation=last_act, kernel_initializer="zeros",
                           use_bias=False, use_sn=use_sn)(VALIDITY)
    CLASS = AdaptiveAveragePooling1D((h * w // 16))(X)
    CLASS = DenseLayer(filter_num_begin,
                       activation=act, use_sn=use_sn)(CLASS)
    CLASS = layers.Flatten()(CLASS)
    CLASS = DenseLayer(last_channel_num,
                       activation='sigmoid', use_sn=use_sn)(CLASS)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[VALIDITY, CLASS])
    return model


def get_swin_class_patch_disc_2d(input_shape, last_channel_num,
                                 filter_num_begin, depth, stack_num_per_depth,
                                 patch_size, stride_mode, num_heads, window_size, num_mlp,
                                 act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    down_ratio = 2 ** depth
    feature_shape = (input_shape[0] // down_ratio,
                     input_shape[1] // down_ratio,
                     -1)
    IN = layers.Input(input_shape)
    X = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    VALIDITY = layers.Reshape(feature_shape)(X)
    # The output section
    VALIDITY = Conv2DLayer(1, kernel_size=3, strides=2,
                           activation=last_act, use_sn=use_sn)(VALIDITY)
    # The output section
    CLASS = layers.GlobalAveragePooling1D()(X)
    CLASS = DenseLayer(last_channel_num,
                       activation='sigmoid', use_sn=use_sn)(CLASS)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[VALIDITY, CLASS])
    return model


def get_swin_class_disc_3d(input_shape, last_channel_num,
                           filter_num_begin, depth, stack_num_per_depth,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False):
    IN = layers.Input(input_shape)
    X = swin_classification_3d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn,
                                    name="classification")
    X = layers.GlobalAveragePooling1D()(X)
    # The output section
    VALIDITY = DenseLayer(1, activation=last_act, use_sn=use_sn)(X)
    # The output section
    CLASS = DenseLayer(
        last_channel_num, activation='sigmoid', use_sn=use_sn)(X)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[VALIDITY, CLASS])
    return model
