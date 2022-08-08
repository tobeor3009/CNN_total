
from __future__ import absolute_import
from sklearn.preprocessing import KernelCenterer

import tensorflow as tf
from .swin_layers import SwinTransformerBlock3D
from tensorflow.keras import layers, backend
from tensorflow.image import extract_patches
from tensorflow import extract_volume_patches


class patch_extract(layers.Layer):
    '''
    Extract patches from the input feature map.

    patches = patch_extract(patch_size)(feature_map)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)

    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`

    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

    '''

    def __init__(self, patch_size, stride_size=None):
        super(patch_extract, self).__init__()
        if stride_size is None:
            stride_size = patch_size
        self.patch_size_row = patch_size[0]
        self.patch_size_col = patch_size[1]
        self.stride_size_row = stride_size[0]
        self.stride_size_col = stride_size[1]

    def call(self, images):

        batch_size = tf.shape(images)[0]
        patches = extract_patches(images=images,
                                  sizes=(1, self.patch_size_row,
                                         self.patch_size_col, 1),
                                  strides=(1, self.stride_size_row,
                                           self.stride_size_col, 1),
                                  rates=(1, 1, 1, 1), padding='SAME')
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size,
                                       patch_num * patch_num,
                                       patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)

        return patches


class patch_extract_3d(layers.Layer):
    '''
    Extract patches from the input feature map.

    patches = patch_extract(patch_size)(feature_map)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)

    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`

    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

    '''

    def __init__(self, patch_size, stride_size=None):
        super(patch_extract_3d, self).__init__()
        if stride_size is None:
            stride_size = patch_size

        self.patch_size_z = patch_size[0]
        self.patch_size_row = patch_size[1]
        self.patch_size_col = patch_size[2]
        self.stride_size_z = patch_size[0]
        self.stride_size_row = stride_size[1]
        self.stride_size_col = stride_size[2]

    def call(self, images):

        batch_size = tf.shape(images)[0]
        patches = extract_volume_patches(images,
                                         ksizes=(1, self.patch_size_z,
                                                 self.patch_size_row,
                                                 self.patch_size_col, 1),
                                         strides=(1, self.stride_size_z,
                                                  self.stride_size_row,
                                                  self.stride_size_col, 1),
                                         padding='SAME')
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)
        patch_dim = patches.shape[-1]
        z_patch_num, row_patch_num, col_patch_num = patches.shape[
            1], patches.shape[2], patches.shape[3]
        patches = tf.reshape(patches, (batch_size,
                                       z_patch_num * row_patch_num * col_patch_num,
                                       patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)

        return patches


class patch_embedding(layers.Layer):
    '''

    Embed patches to tokens.

    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 

    Output
    ----------
        embed: Embedded patches.

    For further information see: https://keras.io/api/layers/core_layers/embedding/

    '''

    def __init__(self, num_patch, embed_dim):
        super(patch_embedding, self).__init__()
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(
            input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        # patch.shape = [B num_patch C]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        # embed.shape = [B num_patch embed_dim] + [num_patch]
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed


class patch_merging(layers.Layer):
    '''
    Downsample embedded patches; it halfs the number of patches
    and double the embedded dimensions (c.f. pooling layers).

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 

    Output
    ----------
        x: downsampled patches.

    '''

    def __init__(self, num_patch, embed_dim, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim

        # A linear transform that doubles the channels
        self.linear_trans = layers.Dense(2 * embed_dim,
                                         use_bias=False,
                                         name='{}_linear_trans'.format(name))

    def call(self, x):

        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 ==
                0), '{}-by-{} patches received, they are not even.'.format(H, W)

        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)

        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H // 2) * (W // 2), 4 * C))

        # Linear transform
        x = self.linear_trans(x)

        return x


class patch_merging_3d(layers.Layer):
    '''
    Downsample embedded patches; it halfs the number of patches
    and double the embedded dimensions (c.f. pooling layers).

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 

    Output
    ----------
        x: downsampled patches.

    '''

    def __init__(self, num_patch, embed_dim, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim

        # A linear transform that doubles the channels
        self.linear_trans = layers.Dense(2 * embed_dim,
                                         use_bias=False,
                                         name='{}_linear_trans'.format(name))

    def call(self, x):

        Z, H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W * Z), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 ==
                0), '{}-by-{} patches received, they are not even.'.format(H, W)

        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, Z, H, W, C))

        # Downsample
        x0 = x[:, :, 0::2, 0::2, :]  # B Z H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B Z H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B Z H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B Z H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)

        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, Z * (H // 2) * (W // 2), 4 * C))

        # Linear transform
        x = self.linear_trans(x)

        return x


class patch_expanding(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.linear_trans1 = layers.Conv2D(upsample_rate * embed_dim,
                                           kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))

        # self.linear_trans2 = layers.Conv2D(upsample_rate * embed_dim,
        #                             kernel_size=1, use_bias=False, name='{}_linear_trans2'.format(name))
        self.prefix = name

    def call(self, x):

        H, W = self.num_patch
        _, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))

        x = self.linear_trans1(x)
        # rearange depth to number of patches
        x = tf.nn.depth_to_space(x, self.upsample_rate,
                                 data_format='NHWC', name='{}_d_to_space'.format(self.prefix))

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1,
                               L * self.upsample_rate * self.upsample_rate,
                               self.embed_dim // 2))
        return x


class Pixelshuffle3D(layers.Layer):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = self.to_tuple(kernel_size)

    # k: kernel, r: resized, o: original
    def call(self, x):
        _, o_h, o_w, o_z, o_c = backend.int_shape(x)
        k_h, k_w, k_z = self.kernel_size
        r_h, r_w, r_z = o_h * k_h, o_w * k_w, o_z * k_z
        r_c = o_c // (k_h * k_w * k_z)

        r_x = layers.Reshape((o_h, o_w, o_z, r_c, k_h, k_w, k_z))(x)
        r_x = layers.Permute((1, 5, 2, 6, 3, 7, 4))(r_x)
        r_x = layers.Reshape((r_h, r_w, r_z, r_c))(r_x)

        return r_x

    def compute_output_shape(self, input_shape):
        _, o_h, o_w, o_z, o_c = input_shape
        k_h, k_w, k_z = self.kernel_size
        r_h, r_w, r_z = o_h * k_h, o_w * k_w, o_z * k_z
        r_c = o_c // (r_h * r_w * r_z)

        return (r_h, r_w, r_z, r_c)

    def to_tuple(self, int_or_tuple):
        if isinstance(int_or_tuple, int):
            convert_tuple = (int_or_tuple, int_or_tuple, int_or_tuple)
        else:
            convert_tuple = int_or_tuple
        return convert_tuple


class patch_expanding_3d(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.linear_trans1 = layers.Conv3D((upsample_rate ** 2) * embed_dim,
                                           kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        self.pixel_shuffle_3d = Pixelshuffle3D(kernel_size=upsample_rate)
        # self.linear_trans2 = layers.Conv2D(upsample_rate * embed_dim,
        #                             kernel_size=1, use_bias=False, name='{}_linear_trans2'.format(name))
        self.prefix = name

    def call(self, x):

        Z, H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W * Z), 'input feature has wrong size'

        x = tf.reshape(x, (-1, Z, H, W, C))

        x = self.linear_trans1(x)

        # rearange depth to number of patches
        x = self.pixel_shuffle_3d(x)

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1,
                               L * (self.upsample_rate ** 3),
                               C // 2))
        return x


class patch_expanding_2d_3d(layers.Layer):

    def __init__(self, num_patch, embed_dim, return_vector=True, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.return_vector = return_vector
        self.embed_dim = embed_dim
        self.linear_trans1 = layers.Conv3D(embed_dim,
                                           kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        # self.linear_trans2 = layers.Conv2D(upsample_rate * embed_dim,
        #                             kernel_size=1, use_bias=False, name='{}_linear_trans2'.format(name))
        self.prefix = name

    def call(self, x):

        H, W = self.num_patch
        Z = H
        B, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, Z, H, W, C // Z))
        x = self.linear_trans1(x)

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1,
                               L * Z,
                               self.embed_dim))
        return x
