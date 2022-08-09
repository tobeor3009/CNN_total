import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers
from . import swin_layers, transformer_layers, utils
from .base_layer import swin_transformer_stack_2d
from .classfication import swin_classification_2d_base

BLOCK_MODE_NAME = "seg"


# Get and Revise From https://github.com/LynnHo/AttGAN-Tensorflow/blob/master/utils.py
def tile_concat(a_list, b_list=[]):
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


def swin_class_gen_2d_base(input_tensor, class_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                           patch_size, stride_mode, num_heads, window_size, num_mlp, act="gelu", shift_window=True,
                           swin_v2=False, name='unet'):
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
    X = transformer_layers.patch_extract(patch_size,
                                         stride_size)(X)
    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x * num_patch_y,
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
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_ - 1):
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
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
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # upsampling begins at the deepest available tensor
    X = tile_concat(X_skip[0], class_tensor)
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                               embed_dim=embed_dim,
                                               upsample_rate=2,
                                               return_vector=True,
                                               swin_v2=swin_v2)(X)
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
                                      name='{}_swin_up{}'.format(name, i))
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    if stride_mode == "half":
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
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
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y),
                                           embed_dim=embed_dim,
                                           upsample_rate=patch_size[0],
                                           return_vector=False,
                                           swin_v2=swin_v2)(X)
    return X


def get_swin_class_gen_2d(input_shape, class_num, last_channel_num,
                          filter_num_begin, depth,
                          stack_num_down, stack_num_up,
                          patch_size, stride_mode, num_heads, window_size, num_mlp,
                          act="gelu", last_act="sigmoid", shift_window=True, swin_v2=False):
    IN = layers.Input(input_shape)
    CLASS = layers.Input(class_num)
    # Base architecture
    X = swin_class_gen_2d_base(IN, CLASS, filter_num_begin, depth, stack_num_down, stack_num_up,
                               patch_size, stride_mode, num_heads, window_size, num_mlp, act=act,
                               shift_window=shift_window, swin_v2=swin_v2, name='unet')

    OUT = layers.Conv2D(last_channel_num, kernel_size=1,
                        use_bias=False, activation=last_act)(X)

    # Model configuration
    model = Model(inputs=[IN, CLASS], outputs=[OUT, ])
    return model


def get_swin_class_disc_2d(input_shape, last_channel_num,
                           filter_num_begin, depth, stack_num_per_depth,
                           patch_size, stride_mode, num_heads, window_size, num_mlp,
                           act="gelu", shift_window=True, swin_v2=False):
    IN = layers.Input(input_shape)
    X = swin_classification_2d_base(IN, filter_num_begin, depth, stack_num_per_depth,
                                    patch_size, stride_mode, num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window, swin_v2=swin_v2, name="classification")
    X = layers.GlobalAveragePooling1D()(X)
    # The output section
    VALIDITY = layers.Dense(1, activation='sigmoid')(X)
    # The output section
    CLASS = layers.Dense(last_channel_num, activation='sigmoid')(X)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[VALIDITY, CLASS])
    return model
