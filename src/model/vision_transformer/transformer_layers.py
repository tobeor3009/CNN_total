
from __future__ import absolute_import

import math
import tensorflow as tf
from tensorflow.keras import layers, backend
from tensorflow.image import extract_patches
from tensorflow import extract_volume_patches
from .util_layers import get_norm_layer
from .util_layers import DenseLayer, Conv2DLayer, Conv3DLayer


class PatchExtract(layers.Layer):
    '''
    Extract patches from the input feature map.

    patches = PatchExtract(patch_size)(feature_map)

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
        super(PatchExtract, self).__init__()
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


class PatchExtract3D(layers.Layer):
    '''
    Extract patches from the input feature map.

    patches = PatchExtract(patch_size)(feature_map)

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
        super(PatchExtract3D, self).__init__()
        if stride_size is None:
            stride_size = patch_size

        self.patch_size_z = patch_size[0]
        self.patch_size_row = patch_size[1]
        self.patch_size_col = patch_size[2]
        self.stride_size_z = stride_size[0]
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


class PatchEmbedding(layers.Layer):
    '''

    Embed patches to tokens.

    patches_embed = PatchEmbedding(num_patch, embed_dim)(pathes)

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

    def __init__(self, num_patch, embed_dim, use_sn=False):
        super(PatchEmbedding, self).__init__()
        self.num_patch = num_patch
        self.proj = DenseLayer(embed_dim, use_sn=use_sn)
        self.pos_embed = layers.Embedding(input_dim=num_patch,
                                          output_dim=embed_dim)

    def call(self, patch):
        # patch.shape = [B num_patch C]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        # embed.shape = [B num_patch embed_dim] + [num_patch]
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed


class PatchMerging(layers.Layer):
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

    def __init__(self, num_patch, embed_dim, norm="layer",
                 swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.swin_v2 = swin_v2
        # A linear transform that doubles the channels
        self.linear_trans = DenseLayer(2 * embed_dim,
                                       use_bias=False, use_sn=use_sn,
                                       name='{}_linear_trans'.format(name))
        self.norm = get_norm_layer(norm)

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
        if self.swin_v2:
            x = self.linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.linear_trans(x)

        return x


class PatchMerging3D(layers.Layer):
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

    def __init__(self, num_patch, embed_dim, include_3d=False, norm="layer",
                 swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.include_3d = include_3d
        self.swin_v2 = swin_v2

        # A linear transform that doubles the channels
        self.linear_trans = DenseLayer(2 * embed_dim,
                                       use_bias=False, use_sn=use_sn,
                                       name='{}_linear_trans'.format(name))
        self.norm = get_norm_layer(norm)

    def call(self, x):

        Z, H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == Z * H * W), 'input feature has wrong size'
        assert (Z % 2 == 0 and H % 2 == 0 and W % 2 ==
                0), '{}-by-{} patches received, they are not even.'.format(H, W)

        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, Z, H, W, C))

        # Downsample
        if self.include_3d:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B Z H/2 W/2 C
            x1 = x[:, 1::2, 0::2, 0::2, :]  # B Z H/2 W/2 C
            x2 = x[:, 0::2, 1::2, 0::2, :]  # B Z H/2 W/2 C
            x3 = x[:, 1::2, 1::2, 0::2, :]  # B Z H/2 W/2 C
            x4 = x[:, 0::2, 0::2, 1::2, :]  # B Z H/2 W/2 C
            x5 = x[:, 1::2, 0::2, 1::2, :]  # B Z H/2 W/2 C
            x6 = x[:, 0::2, 1::2, 1::2, :]  # B Z H/2 W/2 C
            x7 = x[:, 1::2, 1::2, 1::2, :]  # B Z H/2 W/2 C
            x = tf.concat((x0, x1, x2, x3, x4, x5, x6, x7),
                          axis=-1)
            x = tf.reshape(x,
                           shape=(-1, (Z // 2) * (H // 2) * (W // 2), 8 * C))
        else:
            x0 = x[:, :, 0::2, 0::2, :]  # B Z H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B Z H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B Z H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B Z H/2 W/2 C
            x = tf.concat((x0, x1, x2, x3),
                          axis=-1)
            # Convert to the patch squence
            x = tf.reshape(x,
                           shape=(-1, Z * (H // 2) * (W // 2), 4 * C))

        # Linear transform
        if self.swin_v2:
            x = self.linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.linear_trans(x)

        return x


class PatchExpanding(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate,
                 return_vector=True, norm="layer", swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.norm = get_norm_layer(norm)
        self.swin_v2 = swin_v2

        self.upsample_layer = layers.UpSampling2D(size=upsample_rate,
                                                  interpolation="bilinear")
        self.upsample_linear_trans = Conv2DLayer(embed_dim // 2,
                                                 kernel_size=1, use_bias=False, use_sn=use_sn,
                                                 name='{}_upsample_linear_trans'.format(name))
        self.pixel_shuffle_linear_trans = Conv2DLayer(upsample_rate * embed_dim,
                                                      kernel_size=1, use_bias=False, use_sn=use_sn,
                                                      name='{}_pixel_shuffle_linear_trans'.format(name))
        self.concat_linear_trans = Conv2DLayer(embed_dim // 2,
                                               kernel_size=1, use_bias=False, use_sn=use_sn,
                                               name='{}_concat_linear_trans'.format(name))
        self.prefix = name

    def call(self, x):

        H, W = self.num_patch
        _, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))

        upsample = self.upsample_layer(x)
        upsample = self.upsample_linear_trans(upsample)

        pixel_shuffle = self.pixel_shuffle_linear_trans(x)
        # rearange depth to number of patches
        pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle, self.upsample_rate,
                                             data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        x = tf.concat([upsample, pixel_shuffle], axis=-1)

        if self.swin_v2:
            x = self.concat_linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.concat_linear_trans(x)
        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1,
                               L * self.upsample_rate * self.upsample_rate,
                               self.embed_dim // 2))
        return x


class PatchExpandingSimple(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate,
                 return_vector=True, norm="layer", swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.norm = get_norm_layer(norm)
        self.swin_v2 = swin_v2

        self.pixel_shuffle_linear_trans = Conv2DLayer(upsample_rate * embed_dim,
                                                      kernel_size=1, use_bias=False, use_sn=use_sn,
                                                      name='{}_pixel_shuffle_linear_trans'.format(name))
        self.linear_trans = Conv2DLayer(embed_dim // 2,
                                        kernel_size=1, use_bias=False, use_sn=use_sn,
                                        name='{}_concat_linear_trans'.format(name))
        self.prefix = name

    def call(self, x):

        H, W = self.num_patch
        _, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))

        x = self.pixel_shuffle_linear_trans(x)
        # rearange depth to number of patches
        x = tf.nn.depth_to_space(x, self.upsample_rate,
                                 data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        if self.swin_v2:
            x = self.linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.linear_trans(x)
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


class PatchExpanding3D(layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, norm="layer",
                 swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.norm = get_norm_layer(norm)
        self.swin_v2 = swin_v2

        self.upsample_layer = layers.UpSampling3D(size=upsample_rate)
        self.upsample_linear_trans = Conv3DLayer(embed_dim // 2,
                                                 kernel_size=1, use_bias=False, use_sn=use_sn,
                                                 name='{}_upsample_linear_trans'.format(name))
        self.pixel_shuffle_linear_trans = Conv3DLayer((upsample_rate ** 2) * embed_dim,
                                                      kernel_size=1, use_bias=False, use_sn=use_sn,
                                                      name='{}_pixel_shuffle_linear_trans'.format(name))
        self.pixel_shuffle_3d = Pixelshuffle3D(kernel_size=upsample_rate)
        self.concat_linear_trans = Conv3DLayer(embed_dim // 2,
                                               kernel_size=1, use_bias=False, use_sn=use_sn,
                                               name='{}_concat_linear_trans'.format(name))

        self.prefix = name

    def call(self, x):

        Z, H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W * Z), 'input feature has wrong size'

        x = tf.reshape(x, (-1, Z, H, W, C))

        upsample = self.upsample_linear_trans(x)
        upsample = self.upsample_layer(upsample)

        pixel_shuffle = self.pixel_shuffle_linear_trans(x)
        # rearange depth to number of patches
        pixel_shuffle = self.pixel_shuffle_3d(pixel_shuffle)
        x = tf.concat([upsample, pixel_shuffle], axis=-1)
        if self.swin_v2:
            x = self.concat_linear_trans(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.concat_linear_trans(x)

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1,
                               L * (self.upsample_rate ** 3),
                               self.embed_dim // 2))
        return x


class PatchExpanding_2D_3D(layers.Layer):
    def __init__(self, num_patch, embed_dim, return_vector=True, preserve_dim=False, norm="layer",
                 swin_v2=False, use_sn=False, name=''):
        super().__init__()

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.return_vector = return_vector
        self.embed_dim = embed_dim
        self.swin_v2 = swin_v2

        dim_scale = 2

        total_downsample = int(math.log(num_patch[0], 2))
        expand_num = total_downsample // 2
        is_odd = total_downsample % 2
        expand_num += is_odd
        self.downsample_scale_list = []
        self.expand_list = []
        self.norm_list = []
        for idx in range(expand_num):
            if idx < expand_num - 1:
                down_scale = 4
                expand_dim = 4 * embed_dim
                new_dim = embed_dim
            else:
                if is_odd == 1:
                    down_scale = 2
                    expand_dim = embed_dim
                else:
                    down_scale = 4
                    expand_dim = 2 * embed_dim
                new_dim = embed_dim // dim_scale
                if preserve_dim:
                    expand_dim *= 2
                    new_dim *= 2
            expand_layer = DenseLayer(expand_dim,
                                      use_sn=use_sn, use_bias=False)
            norm_layer = get_norm_layer(norm)
            self.downsample_scale_list.append(down_scale)
            self.expand_list.append(expand_layer)
            self.norm_list.append(norm_layer)

        self.prefix = name

    def call(self, x):

        H, W = self.num_patch
        Z = H
        B, L, C = x.get_shape().as_list()
        assert (L == H * W), 'input feature has wrong size'

        for down_scale, expand_layer, norm_layer in zip(self.downsample_scale_list, self.expand_list, self.norm_list):
            x = expand_layer(x)
            B, L, C = x.get_shape().as_list()
            x = layers.Reshape((-1, H, W, C))(x)
            # x.shape = [B, Z, H, W, C] => [B, Z, C, H, W]
            x = tf.transpose(x, (0, 1, 4, 2, 3))
            x = layers.Reshape((-1, C // down_scale, H, W))(x)
            # x.shape = [B, Z, C, H, W] => [B, Z, H, W, C]
            x = tf.transpose(x, (0, 1, 3, 4, 2))
            x = layers.Reshape((-1, C // down_scale))(x)
            x = norm_layer(x)

        if self.return_vector:
            pass
        else:
            x = layers.Reshape((Z, H, W, C // down_scale))(x)
        return x
