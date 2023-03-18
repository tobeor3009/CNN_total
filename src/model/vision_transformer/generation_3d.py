import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import AdaptiveAveragePooling1D, AdaptiveAveragePooling2D, AdaptiveAveragePooling3D
from . import swin_layers, transformer_layers, utils
from .base_layer import swin_transformer_stack_3d, swin_context_transformer_stack_2d
from .classfication import swin_classification_2d_base, swin_classification_3d_base, get_swin_classification_2d
from .util_layers import DenseLayer, Conv1DLayer, Conv2DLayer, Conv3DLayer, get_act_layer
from tensorflow.keras.models import clone_model
from tensorflow_addons.layers import InstanceNormalization
from copy import deepcopy
from .generation import swin_class_gen_2d_base_v3
from .base_layer import SwinTransformerStack2D
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


def tile_concat_3d(a_list, b_list=[]):
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


def swin_class_gen_3d_base_v3(input_tensor, class_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                              patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                              include_3d=True, swin_v2=False, use_sn=False, name='unet', add_class_info_position=["middle"]):
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
    # Embed patches to tokens
    X = transformer_layers.PatchEmbedding(num_patch_z * num_patch_x * num_patch_y,
                                          embed_dim, use_sn=use_sn)(X)
    # The first Swin Transformer stack
    X = swin_transformer_stack_3d(X,
                                  stack_num=stack_num_down,
                                  embed_dim=embed_dim,
                                  num_patch=(
                                      num_patch_z, num_patch_x, num_patch_y),
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
        if "encode" in add_class_info_position:
            X = tile_concat_1d(X, class_tensor)
        X = transformer_layers.PatchMerging3D((num_patch_z, num_patch_x, num_patch_y),
                                              embed_dim=embed_dim,
                                              include_3d=include_3d,
                                              swin_v2=swin_v2,
                                              use_sn=use_sn,
                                              name='down{}'.format(i))(X)

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_z //= 2
        num_patch_x //= 2
        num_patch_y //= 2
        # Swin Transformer stacks
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_down,
                                      embed_dim=embed_dim,
                                      num_patch=(
                                          num_patch_z, num_patch_x, num_patch_y),
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
    if "middle" in add_class_info_position:
        X = tile_concat_1d(X_skip[0], class_tensor)
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    for i in range(depth_decode):
        if "decode" in add_class_info_position and i > 0:
            X = tile_concat_1d(X, class_tensor)
        # Patch expanding
        X = transformer_layers.PatchExpanding3DSimple(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                                      embed_dim=embed_dim,
                                                      upsample_rate=2,
                                                      return_vector=True,
                                                      swin_v2=swin_v2,
                                                      use_sn=use_sn,
                                                      name=f'{name}_swin_expanding_{i}')(X)
        # update token shape info
        embed_dim = embed_dim // 2
        num_patch_z *= 2
        num_patch_x *= 2
        num_patch_y *= 2
        # Concatenation and linear projection
        X = layers.concatenate([X, X_decode[i]], axis=-1,
                               name='{}_concat_{}'.format(name, i))
        X = DenseLayer(embed_dim, use_bias=False, use_sn=use_sn,
                       name='{}_concat_linear_proj_{}'.format(name, i))(X)

        # Swin Transformer stacks
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(
                                          num_patch_z, num_patch_x, num_patch_y),
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
        X = transformer_layers.PatchExpanding3DSimple((num_patch_z, num_patch_x, num_patch_y),
                                                      embed_dim=embed_dim, use_sn=use_sn,
                                                      name='down_last')(X)
        embed_dim *= 2
        num_patch_z //= 2
        num_patch_x //= 2
        num_patch_y //= 2
        X = swin_transformer_stack_3d(X,
                                      stack_num=stack_num_up,
                                      embed_dim=embed_dim,
                                      num_patch=(
                                          num_patch_z, num_patch_x, num_patch_y),
                                      num_heads=num_heads[i],
                                      window_size=window_size[i],
                                      num_mlp=num_mlp,
                                      act=act,
                                      shift_window=shift_window,
                                      mode=BLOCK_MODE_NAME,
                                      swin_v2=swin_v2,
                                      use_sn=use_sn,
                                      name='{}_swin_down_last'.format(name))
    X = transformer_layers.PatchExpanding3DSimple(num_patch=(num_patch_z, num_patch_x, num_patch_y),
                                                  embed_dim=embed_dim,
                                                  upsample_rate=patch_size,
                                                  return_vector=False,
                                                  swin_v2=swin_v2,
                                                  name=f'{name}_swin_expanding_last')(X)
    return X


def swin_seg_disc_v3(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                     patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                     include_3d=True, swin_v2=False, use_sn=False, last_act=None, class_num=1, name='unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head

    '''
    _, Z, H, W, _ = input_tensor.shape
    z = Z // (2 ** depth) // patch_size[0]
    h = H // (2 ** depth) // patch_size[1]
    w = W // (2 ** depth) // patch_size[2]
    # Compute number be patches to be embeded
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_z, num_patch_x, num_patch_y = utils.get_image_patch_num_3d(input_size[:-1],
                                                                         patch_size,
                                                                         stride_size)
    num_patch_array = np.array([num_patch_z, num_patch_x, num_patch_y])
    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    depth_ = depth

    # Patch extraction
    FRONT_X = transformer_layers.PatchExtract3D(patch_size,
                                                stride_size)(input_tensor)
    # Embed patches to tokens
    FRONT_X = transformer_layers.PatchEmbedding(num_patch_array.prod(),
                                                embed_dim, use_sn=use_sn)(FRONT_X)
    # The first Swin Transformer stack
    FRONT_X = swin_transformer_stack_3d(FRONT_X,
                                        stack_num=stack_num_down,
                                        embed_dim=embed_dim,
                                        num_patch=num_patch_array,
                                        num_heads=num_heads[0],
                                        window_size=window_size[0],
                                        num_mlp=num_mlp,
                                        act=act,
                                        mode="classification",
                                        shift_window=shift_window,
                                        swin_v2=swin_v2,
                                        use_sn=use_sn,
                                        name='{}_swin_down'.format(name))
    i = 0
    FRONT_X = transformer_layers.PatchMerging3D(num_patch_array,
                                                embed_dim=embed_dim,
                                                include_3d=include_3d,
                                                swin_v2=swin_v2,
                                                use_sn=use_sn,
                                                name='down{}'.format(i))(FRONT_X)
    # update token shape info
    embed_dim = embed_dim * 2
    num_patch_array = num_patch_array // 2

    # Swin Transformer stacks
    FRONT_X = swin_transformer_stack_3d(FRONT_X,
                                        stack_num=stack_num_down,
                                        embed_dim=embed_dim,
                                        num_patch=num_patch_array,
                                        num_heads=num_heads[i + 1],
                                        window_size=window_size[i + 1],
                                        num_mlp=num_mlp,
                                        act=act,
                                        shift_window=shift_window,
                                        mode="classification",
                                        swin_v2=swin_v2,
                                        use_sn=use_sn,
                                        name='{}_swin_down{}'.format(name, i + 1))

    VALIDITY = FRONT_X
    LABEL = FRONT_X
    # Downsampling blocks
    for i in range(1, depth_ - 1):
        # Patch merging
        VALIDITY = transformer_layers.PatchMerging3D(num_patch_array,
                                                     embed_dim=embed_dim,
                                                     include_3d=include_3d,
                                                     swin_v2=swin_v2,
                                                     use_sn=use_sn,
                                                     name='down_validity{}'.format(i))(VALIDITY)
        LABEL = transformer_layers.PatchMerging3D(num_patch_array,
                                                  embed_dim=embed_dim,
                                                  include_3d=include_3d,
                                                  swin_v2=swin_v2,
                                                  use_sn=use_sn,
                                                  name='down_label{}'.format(i))(LABEL)

        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_array = num_patch_array // 2

        # Swin Transformer stacks
        VALIDITY = swin_transformer_stack_3d(VALIDITY,
                                             stack_num=stack_num_down,
                                             embed_dim=embed_dim,
                                             num_patch=num_patch_array,
                                             num_heads=num_heads[i + 1],
                                             window_size=window_size[i + 1],
                                             num_mlp=num_mlp,
                                             act=act,
                                             shift_window=shift_window,
                                             mode="classification",
                                             swin_v2=swin_v2,
                                             use_sn=use_sn,
                                             name='{}_swin_down_validity{}'.format(name, i + 1))
        LABEL = swin_transformer_stack_3d(LABEL,
                                          stack_num=stack_num_down,
                                          embed_dim=embed_dim,
                                          num_patch=num_patch_array,
                                          num_heads=num_heads[i + 1],
                                          window_size=window_size[i + 1],
                                          num_mlp=num_mlp,
                                          act=act,
                                          shift_window=shift_window,
                                          mode="classification",
                                          swin_v2=swin_v2,
                                          use_sn=use_sn,
                                          name='{}_swin_down_label{}'.format(name, i + 1))
    return VALIDITY, LABEL


class SwinClassGen2DBaseV3(layers.Layer):
    def __init__(self, input_shape, filter_num_begin, depth, stack_num_down, stack_num_up,
                 patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                 swin_v2=False, use_sn=False, name='unet', return_vector=False, add_class_info_position=["middle"]):
        super(SwinClassGen2DBaseV3, self).__init__()

        if stride_mode == "same":
            stride_size = patch_size
        elif stride_mode == "half":
            stride_size = np.array(patch_size) // 2
        self.input_row, self.input_col = input_shape[:-1]
        self.filter_num_begin = filter_num_begin
        self.add_class_info_position = add_class_info_position
        num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_shape[:-1],
                                                                patch_size,
                                                                stride_size)
        # Number of Embedded dimensions
        embed_dim = filter_num_begin

        depth_ = depth
        self.patch_extract_layer = transformer_layers.PatchExtract(patch_size,
                                                                   stride_size)
        self.patch_embedding_layer = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                                                       embed_dim, use_sn=use_sn)
        self.first_stack_layer = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                        num_patch=(
                                                            num_patch_x, num_patch_y),
                                                        num_heads=num_heads[0],
                                                        window_size=window_size[0],
                                                        num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                        swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down'.format(name))

        self.encoder_merge_layer_list = []
        self.encoder_stack_layer_list = []
        for i in range(depth_ - 1):
            encoder_merge_layer = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                                                  embed_dim=embed_dim,
                                                                  swin_v2=swin_v2,
                                                                  use_sn=use_sn,
                                                                  name='down{}'.format(i))
            # update token shape info
            embed_dim *= 2
            num_patch_x //= 2
            num_patch_y //= 2

            # Swin Transformer stacks
            encoder_stack_layer = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                         num_patch=(num_patch_x,
                                                                    num_patch_y),
                                                         num_heads=num_heads[i + 1],
                                                         window_size=window_size[i + 1],
                                                         num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                         swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down{}'.format(name, i + 1))
            self.encoder_merge_layer_list.append(encoder_merge_layer)
            self.encoder_stack_layer_list.append(encoder_stack_layer)

        num_heads = num_heads[::-1]
        window_size = window_size[::-1]
        self.skip_concat_layer = layers.Concatenate(axis=-1)
        self.decoder_expand_layer_list = []
        self.decoder_skip_dense_layer_list = []
        self.decoder_stack_layer_list = []
        for i in range(depth_ - 1):
            decoder_expand_layer = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                                                     embed_dim=embed_dim,
                                                                     upsample_rate=2,
                                                                     return_vector=True,
                                                                     swin_v2=swin_v2,
                                                                     use_sn=use_sn,
                                                                     name=f'{name}_swin_expanding_{i}')
            embed_dim //= 2
            num_patch_x *= 2
            num_patch_y *= 2
            decoder_skip_dense_layer = DenseLayer(embed_dim, use_bias=False, use_sn=use_sn,
                                                  name='{}_concat_linear_proj_{}'.format(name, i))
            decoder_stack_layer = SwinTransformerStack2D(stack_num=stack_num_up, embed_dim=embed_dim,
                                                         num_patch=(num_patch_x,
                                                                    num_patch_y),
                                                         num_heads=num_heads[i],
                                                         window_size=window_size[i],
                                                         num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                         swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down'.format(name))
            self.decoder_expand_layer_list.append(decoder_expand_layer)
            self.decoder_skip_dense_layer_list.append(decoder_skip_dense_layer)
            self.decoder_stack_layer_list.append(decoder_stack_layer)
        self.last_expand_layer = transformer_layers.PatchExpanding(num_patch=(num_patch_x, num_patch_y),
                                                                   embed_dim=embed_dim,
                                                                   upsample_rate=patch_size[0],
                                                                   return_vector=return_vector,
                                                                   swin_v2=swin_v2,
                                                                   name=f'{name}_swin_expanding_last')

    def call(self, input_list):
        input_tensor, class_tensor = input_list
        skip_x_list = []
        x = self.patch_extract_layer(input_tensor)
        x = self.patch_embedding_layer(x)
        x = self.first_stack_layer(x)
        skip_x_list.append(x)
        for encoder_merge_layer, encoder_stack_layer in zip(self.encoder_merge_layer_list,
                                                            self.encoder_stack_layer_list):
            if "encoder" in self.add_class_info_position:
                x = tile_concat_1d(x, class_tensor)
            x = encoder_merge_layer(x)
            x = encoder_stack_layer(x)
            skip_x_list.append(x)
        skip_x_list = skip_x_list[::-1]
        x = skip_x_list.pop(0)
        if "middle" in self.add_class_info_position:
            x = tile_concat_1d(x, class_tensor)

        for i, (decoder_expand_layer, decoder_skip_dense_layer, decoder_stack_layer) in enumerate(zip(self.decoder_expand_layer_list,
                                                                                                      self.decoder_skip_dense_layer_list,
                                                                                                      self.decoder_stack_layer_list)):
            if "decode" in self.add_class_info_position and i > 0:
                x = tile_concat_1d(x, class_tensor)
            x = decoder_expand_layer(x)
            x = self.skip_concat_layer([x, skip_x_list[i]])
            x = decoder_skip_dense_layer(x)
            x = decoder_stack_layer(x)
        x = self.last_expand_layer(x)

        return x

    def compute_output_shape(self, input_shape_list):
        image_input_shape, class_input_shape = input_shape_list
        output_shape = tensor_shape.TensorShape(
            image_input_shape[:-1] + [self.filter_num_begin // 2]
        )
        return output_shape


class SwinDisc2DBaseV3(layers.Layer):
    def __init__(self, input_shape, filter_num_begin, depth, stack_num_down,
                 patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                 swin_v2=False, use_sn=False, name='unet'):
        super(SwinDisc2DBaseV3, self).__init__()

        if stride_mode == "same":
            stride_size = patch_size
        elif stride_mode == "half":
            stride_size = np.array(patch_size) // 2
        self.input_row, self.input_col = input_shape[:-1]
        self.filter_num_begin = filter_num_begin
        self.depth = depth
        self.patch_size = patch_size
        num_patch_x, num_patch_y = utils.get_image_patch_num_2d(input_shape[:-1],
                                                                patch_size,
                                                                stride_size)
        # Number of Embedded dimensions
        embed_dim = filter_num_begin

        depth_ = depth
        self.patch_extract_layer = transformer_layers.PatchExtract(patch_size,
                                                                   stride_size)
        self.patch_embedding_layer = transformer_layers.PatchEmbedding(num_patch_x * num_patch_y,
                                                                       embed_dim, use_sn=use_sn)
        self.stack_layer_1 = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                    num_patch=(num_patch_x,
                                                               num_patch_y),
                                                    num_heads=num_heads[0],
                                                    window_size=window_size[0],
                                                    num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                    swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down'.format(name))

        i = 0
        self.merge_layer_1 = transformer_layers.PatchMerging(num_patch=(num_patch_x,
                                                                        num_patch_y),
                                                             embed_dim=embed_dim,
                                                             swin_v2=swin_v2,
                                                             use_sn=use_sn,
                                                             name='down{}'.format(i))
        # update token shape info
        embed_dim = embed_dim * 2
        num_patch_x //= 2
        num_patch_y //= 2

        self.stack_layer_2 = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                    num_patch=(num_patch_x,
                                                               num_patch_y),
                                                    num_heads=num_heads[i + 1],
                                                    window_size=window_size[i + 1],
                                                    num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                    swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down{}'.format(name, i + 1))

        self.validity_merge_layer_list = []
        self.validity_stack_layer_list = []
        self.label_merge_layer_list = []
        self.label_stack_layer_list = []
        for i in range(1, depth_ - 1):
            validity_merge_layer = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                                                   embed_dim=embed_dim,
                                                                   swin_v2=swin_v2,
                                                                   use_sn=use_sn,
                                                                   name='down_validity{}'.format(i + 1))
            label_merge_layer = transformer_layers.PatchMerging((num_patch_x, num_patch_y),
                                                                embed_dim=embed_dim,
                                                                swin_v2=swin_v2,
                                                                use_sn=use_sn,
                                                                name='down_label{}'.format(i + 1))

            # update token shape info
            embed_dim *= 2
            num_patch_x //= 2
            num_patch_y //= 2

            # Swin Transformer stacks
            validity_stack_layer = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                          num_patch=(num_patch_x,
                                                                     num_patch_y),
                                                          num_heads=num_heads[i + 1],
                                                          window_size=window_size[i + 1],
                                                          num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                          swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down_validity{}'.format(name, i + 1))
            label_stack_layer = SwinTransformerStack2D(stack_num=stack_num_down, embed_dim=embed_dim,
                                                       num_patch=(num_patch_x,
                                                                  num_patch_y),
                                                       num_heads=num_heads[i + 1],
                                                       window_size=window_size[i + 1],
                                                       num_mlp=num_mlp, act=act, shift_window=shift_window, mode=BLOCK_MODE_NAME,
                                                       swin_v2=swin_v2, use_sn=use_sn, name='{}_swin_down_label{}'.format(name, i + 1))
            self.validity_merge_layer_list.append(validity_merge_layer)
            self.label_merge_layer_list.append(label_merge_layer)
            self.validity_stack_layer_list.append(validity_stack_layer)
            self.label_stack_layer_list.append(label_stack_layer)

    def call(self, input_tensor):
        x = self.patch_extract_layer(input_tensor)
        x = self.patch_embedding_layer(x)
        x = self.stack_layer_1(x)
        x = self.merge_layer_1(x)
        x = self.stack_layer_2(x)
        validity = x
        label = x
        for validity_merge_layer, label_merge_layer, validity_stack_layer, label_stack_layer in zip(self.validity_merge_layer_list,
                                                                                                    self.label_merge_layer_list,
                                                                                                    self.validity_stack_layer_list,
                                                                                                    self.label_stack_layer_list):
            validity = validity_merge_layer(validity)
            validity = validity_stack_layer(validity)
            label = label_merge_layer(label)
            label = label_stack_layer(label)
        return validity, label

    def compute_output_shape(self, input_shape):

        row_num = self.input_row // (2 ** self.depth) // self.patch_size[0] * 2
        col_num = self.input_col // (2 ** self.depth) // self.patch_size[1] * 2
        last_filter_num = self.filter_num_begin * (2 ** (self.depth - 1))
        output_shape = tensor_shape.TensorShape(
            input_shape[0:1] + [row_num * col_num, last_filter_num]
        )
        return (output_shape, output_shape)


def get_seg_swin_disc_3d_v5(input_shape, class_num,
                            filter_num_begin, depth, stack_num_down, stack_num_up,
                            patch_size, stride_mode, num_heads, window_size, num_mlp,
                            act="gelu", last_act="sigmoid", shift_window=True, include_3d=True, swin_v2=False, use_sn=False):
    Z, H, W, _ = input_shape
    z, h, w = Z // (2 ** depth), H // (2 ** depth), W // (2 ** depth)
    IN = layers.Input(input_shape)
    # Base architecture
    VALIDITY, LABEL = swin_seg_disc_v3(IN, filter_num_begin, depth, stack_num_down, stack_num_up,
                                       patch_size, stride_mode, num_heads, window_size, num_mlp,
                                       act=act, shift_window=shift_window, include_3d=include_3d,
                                       swin_v2=swin_v2, use_sn=use_sn, name="label")

    VALIDITY = layers.Reshape((z, h, w, -1))(VALIDITY)
    VALIDITY = Conv3DLayer(filter_num_begin, 3, padding="same",
                           activation=act, use_sn=use_sn)(VALIDITY)
    VALIDITY = AdaptiveAveragePooling3D((8, 8, 8))(VALIDITY)
    VALIDITY = Conv3DLayer(filter_num_begin, 3, padding="same",
                           activation=act, use_sn=use_sn)(VALIDITY)
    VALIDITY = Conv3DLayer(1, 1,
                           activation=last_act, use_sn=use_sn)(VALIDITY)
    LABEL = layers.Flatten()(LABEL)
    LABEL = AdaptiveAveragePooling1D(filter_num_begin * 4)(LABEL)
    LABEL = DenseLayer(filter_num_begin, activation=act, use_sn=use_sn)(LABEL)
    LABEL = layers.Flatten()(LABEL)
    LABEL = DenseLayer(class_num, activation='sigmoid', use_sn=use_sn)(LABEL)
    model = Model(inputs=[IN], outputs=[VALIDITY, LABEL])
    return model


def get_seg_swin_disc_3d_by_2d_v5(input_shape, class_num,
                                  filter_num_begin, depth, stack_num_down, stack_num_up,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp,
                                  act="gelu", last_act="sigmoid", shift_window=True, include_3d=True, swin_v2=False, use_sn=False):
    Z, H, W, _ = input_shape
    z = Z
    h = H // (2 ** depth) // patch_size[1] * 2
    w = W // (2 ** depth) // patch_size[2] * 2
    feature_dim = filter_num_begin * (2 ** (depth - 1))
    IN = layers.Input(input_shape)
    # Base architecture
    feature_2d_layer = SwinDisc2DBaseV3(input_shape[1:], filter_num_begin, depth, stack_num_down,
                                        patch_size[1:], stride_mode, num_heads, window_size, num_mlp, act, shift_window,
                                        swin_v2, use_sn, name="feature_2d")
    VALIDITY, LABEL = layers.TimeDistributed(feature_2d_layer)(IN)
    VALIDITY = layers.Reshape((z, h, w, feature_dim))(VALIDITY)
    VALIDITY = Conv3DLaye`r(filter_num_begin, 3, padding="same",
                           activation=act, use_sn=use_sn, name="validity_conv_1")(VALIDITY)
    VALIDITY = AdaptiveAveragePooling3D((8, 8, 8))(VALIDITY)
    VALIDITY = Conv3DLayer(filter_num_begin, 3, padding="same",
                           activation=act, use_sn=use_sn, name="validity_conv_2")(VALIDITY)
    VALIDITY = Conv3DLayer(1, 1,
                           activation=last_act, use_sn=use_sn, name="validity_conv_3")(VALIDITY)
    LABEL = layers.Flatten()(LABEL)
    LABEL = AdaptiveAveragePooling1D(filter_num_begin * 4)(LABEL)
    LABEL = DenseLayer(filter_num_begin, activation=act,
                       use_sn=use_sn, name="label_dense_1")(LABEL)
    LABEL = layers.Flatten()(LABEL)
    LABEL = DenseLayer(class_num, activation='sigmoid',
                       use_sn=use_sn, name="label_dense_1")(LABEL)
    model = Model(inputs=[IN], outputs=[VALIDITY, LABEL])
    return model


def get_swin_class_gen_3d_v3(input_shape, class_num, last_channel_num,
                             filter_num_begin, depth,
                             stack_num_down, stack_num_up,
                             patch_size, stride_mode, num_heads, window_size, num_mlp,
                             act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False,
                             include_3d=True, add_class_info_position=["middle"]):
    IN = layers.Input(input_shape)
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    OUT_CLASS = layers.Input(class_num)
    # Base architecture
    X = swin_class_gen_3d_base_v3(IN, OUT_CLASS, filter_num_begin, depth, stack_num_down, stack_num_up,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                                  shift_window=shift_window, swin_v2=swin_v2, use_sn=use_sn, name='unet',
                                  include_3d=include_3d, add_class_info_position=add_class_info_position)
    OUT = Conv3DLayer(last_channel_num, kernel_size=1,
                      use_bias=False, activation=last_act, use_sn=use_sn)(X)

    # Model configuration
    model = Model(inputs=[IN, OUT_CLASS], outputs=[OUT, ])
    return model


def get_swin_class_gen_3d_by_2d_v3(input_shape, class_num, last_channel_num,
                                   filter_num_begin, depth,
                                   stack_num_down, stack_num_up,
                                   patch_size, stride_mode, num_heads, window_size, num_mlp,
                                   act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False, use_sn=False,
                                   include_3d=True, add_class_info_position=["middle"]):
    if stride_mode == "same":
        stride_size = patch_size
    elif stride_mode == "half":
        stride_size = np.array(patch_size) // 2
    input_shape = np.array(input_shape)
    z_num = input_shape[0]
    model_input = layers.Input(input_shape)
    model_class_input = layers.Input(class_num)
    feature_2d_layer = SwinClassGen2DBaseV3(input_shape[1:], filter_num_begin, depth, stack_num_down, stack_num_up,
                                            patch_size[1:], stride_mode, num_heads, window_size, num_mlp, act, shift_window,
                                            swin_v2, use_sn, name="feature_2d", return_vector=False,
                                            add_class_info_position=add_class_info_position)
    feature_3d_by_2d = layers.TimeDistributed(feature_2d_layer)([model_input,
                                                                 tf.tile(model_class_input[:, None], [1, z_num, 1])])

    feature_3d = Conv3DLayer(filter_num_begin // 2, kernel_size=4, padding="same",
                             use_bias=False, activation=None, use_sn=use_sn, name="conv3d_1")(feature_3d_by_2d)
    feature_3d = InstanceNormalization(axis=-1)(feature_3d)
    feature_3d = get_act_layer(act)(feature_3d)
    # feature_3d = layers.Reshape((input_shape[:-1].prod(),
    #                              filter_num_begin // 2))(feature_3d)
    # feature_3d = swin_transformer_stack_3d(feature_3d,
    #                                        stack_num=stack_num_down,
    #                                        embed_dim=filter_num_begin // 2,
    #                                        num_patch=input_shape[:-1],
    #                                        num_heads=num_heads[0],
    #                                        window_size=window_size[-1],
    #                                        num_mlp=num_mlp,
    #                                        act=act,
    #                                        mode=BLOCK_MODE_NAME,
    #                                        shift_window=shift_window,
    #                                        swin_v2=swin_v2,
    #                                        use_sn=use_sn,
    #                                        name='{}_swin_down'.format("last"))
    # feature_3d = layers.Reshape((*input_shape[:-1],
    #                              filter_num_begin // 2))(feature_3d)
    output = Conv3DLayer(last_channel_num, kernel_size=1,
                         use_bias=False, activation=last_act, use_sn=use_sn, name="conv3d_2")(feature_3d)

    # Model configuration
    model = Model(inputs=[model_input, model_class_input], outputs=[output, ])
    return model
