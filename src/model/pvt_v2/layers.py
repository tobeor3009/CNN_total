import math

import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers


dense_init = initializers.TruncatedNormal(mean=0, stddev=0.02)
dense_bias_init = initializers.Zeros()
identity_layer = layers.Lambda(lambda x: x)


def to2_tuple(int_or_tuple):
    if isinstance(int_or_tuple, tuple):
        return int_or_tuple
    elif isinstance(int_or_tuple, int):
        return (int_or_tuple, int_or_tuple)


def custom_init_dense(units, use_bias=True):

    return layers.Dense(units,
                        use_bias=use_bias,
                        kernel_initializer=dense_init,
                        bias_initializer=dense_bias_init)


def custom_init_layer_norm(epsilon=1e-4):

    return layers.LayerNormalization(axis=-1,
                                     epsilon=epsilon,
                                     beta_initializer='zeros',
                                     gamma_initializer='ones')


def custom_init_conv2d(dim, kernel_size, stride, groups=1):
    kernel_size = to2_tuple(kernel_size)
    fan_out = kernel_size[0] * kernel_size[1] * dim
    fan_out //= groups
    init_stddev = math.sqrt(fan_out)
    conv_init = initializers.TruncatedNormal(mean=0.0, stddev=init_stddev)
    conv_bias_init = initializers.Zeros()
    return layers.Conv2D(filters=dim,
                         kernel_size=kernel_size,
                         padding='same',
                         strides=stride,
                         groups=groups,
                         kernel_initializer=conv_init,
                         bias_initializer=conv_bias_init)


class LayerArchive:
    def __init__(self):
        pass


class TensorArchive:
    def __init__(self):
        pass


def drop_path_(inputs, drop_prob, is_training):

    # Bypass in non-training mode
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    input_shape = tf.shape(inputs)
    batch_num = input_shape[0]
    rank = len(input_shape)

    shape = (batch_num,) + (1,) * (rank - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    path_mask = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * path_mask
    return output


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, training=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training

    def call(self, x):
        return drop_path_(x, self.drop_prob, self.training)


class EncoderMlp(layers.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=activations.gelu,
                 drop=0.,
                 linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = custom_init_dense(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = custom_init_dense(out_features)
        self.drop = layers.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = layers.ReLU()

    def call(self, inputs, H, W):
        x = self.fc1(inputs)
        if self.linear is True:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DecoderMlp(EncoderMlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, hidden, current, H, W):
        inputs = layers.Concatenate(axis=-1)([hidden, current])
        x = self.fc1(inputs)
        if self.linear is True:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class SelfAttention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = custom_init_dense(dim, use_bias=qkv_bias)
        self.kv = custom_init_dense(dim * 2, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = custom_init_dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = custom_init_conv2d(dim=dim,
                                             kernel_size=sr_ratio,
                                             stride=sr_ratio)
                self.norm = custom_init_layer_norm()
        else:
            # in pytorch, nn.AdaptiveAvgPool2d(7)
            self.pool = layers.AveragePooling2D(pool_size=(2, 2),
                                                padding='same')
            self.sr = custom_init_conv2d(dim=dim,
                                         kernel_size=1,
                                         stride=1)
            self.norm = custom_init_layer_norm()
            self.act = activations.gelu

    def build(self, input_shape):
        self.N = input_shape[1]
        self.Down_N = self.N // (self.sr_ratio ** 2)
        self.C = input_shape[2]

    def call(self, x, H, W):
        # x.shape => B, N, C
        # N = H * W
        # Down_N = H * W // (sr_ratio ** 2)
        # q.shape: B, N, dim
        q = self.q(x)
        # q.shape: B, N, self.num_heads, (self.dim // self.num_heads)
        q = layers.Reshape(
            (self.N, self.num_heads, self.dim // self.num_heads))(q)
        # q.shape: B, self.num_heads, N, (self.dim // self.num_heads)
        q = keras_backend.permute_dimensions(q, (0, 2, 1, 3))
        if self.sr_ratio > 1:
            # x.shape: B, N, self.C => x_.shape: B, H, W, self.C
            x_ = layers.Reshape((H, W, self.C))(x)
            # x_shape: B, H, W, dim
            x_ = self.sr(x_)
            # shape: B, (H * W) // (sr_ratio ** 2), dim
            x_ = layers.Reshape((self.Down_N, self.dim))(x_)
            x_ = self.norm(x_)
            if self.linear is True:
                x_ = self.act(x_)
            # shape: B, (H * W) // (sr_ratio ** 2), (dim * 2)
            kv = self.kv(x_)
        else:
            # shape: B, N, (dim * 2) <=> B, (H * W), (dim * 2)
            kv = self.kv(x)

        # shape: B, self.Down_N, (dim * 2) => B, self.Down_N, 2, self.num_heads, (self.dim // num_heads)
        kv = layers.Reshape(
            (self.Down_N, 2, self.num_heads, self.dim // self.num_heads))(kv)
        # shape: 2, B, self.num_heads, self.Down_N, (self.dim // num_heads)
        kv = keras_backend.permute_dimensions(kv, (2, 0, 3, 1, 4))

        # k shape: B, self.num_heads, (self.dim // num_heads), self.Down_N
        # v shape: B, self.num_heads, self.Down_N, (self.dim // num_heads)
        k = keras_backend.permute_dimensions(kv[0], (0, 1, 3, 2))
        v = kv[1]
        # matmul shape: (B, self.num_heads, N, (self.dim // self.num_heads)) @ (B, self.num_heads, (self.dim // num_heads), self.Down_N)
        # attn.shape: B, self.num_heads, N, self.Down_N
        attn = tf.matmul(q, k) * self.scale
        attn = activations.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # matmul shape: (B, self.num_heads, N, self.Down_N) @ (B, self.num_heads, self.Down_N, (self.dim // num_heads)
        # out.shape: B, self.num_heads, N, (self.dim // num_heads)
        out = tf.matmul(attn, v)
        # shape: B, N, self.num_heads, (self.dim // num_heads)
        out = keras_backend.permute_dimensions(out, (0, 2, 1, 3))
        # shape: B, N, dim
        out = layers.Reshape((self.N, self.dim))(out)
        # shape: B, N, dim
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Attention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.N = input_shape[1]
        self.Down_N = self.N // (self.sr_ratio ** 2)
        self.C = input_shape[2]

    def call(self, hidden, current, H, W):
        # x.shape => B, N, C
        # N = H * W
        # Down_N = H * W // (sr_ratio ** 2)
        # q.shape: B, N, dim
        q = self.q(current)
        # q.shape: B, N, self.num_heads, (self.dim // self.num_heads)
        q = layers.Reshape(
            (self.N, self.num_heads, self.dim // self.num_heads))(q)
        # q.shape: B, self.num_heads, N, (self.dim // self.num_heads)
        q = keras_backend.permute_dimensions(q, (0, 2, 1, 3))
        if self.sr_ratio > 1:
            # x.shape: B, N, self.C => x_.shape: B, H, W, self.C
            hidden_small = layers.Reshape((H, W, self.C))(hidden)
            # x_shape: B, H, W, dim
            hidden_small = self.sr(hidden_small)
            # shape: B, (H * W) // (sr_ratio ** 2), dim
            hidden_small = layers.Reshape(
                (self.Down_N, self.dim))(hidden_small)
            hidden_small = self.norm(hidden_small)
            if self.linear is True:
                hidden_small = self.act(hidden_small)
            # shape: B, (H * W) // (sr_ratio ** 2), (dim * 2)
            kv = self.kv(hidden_small)
        else:
            # shape: B, N, (dim * 2) <=> B, (H * W), (dim * 2)
            kv = self.kv(hidden)

        # shape: B, self.Down_N, (dim * 2) => B, self.Down_N, 2, self.num_heads, (self.dim // num_heads)
        kv = layers.Reshape(
            (self.Down_N, 2, self.num_heads, self.dim // self.num_heads))(kv)
        # shape: 2, B, self.num_heads, self.Down_N, (self.dim // num_heads)
        kv = keras_backend.permute_dimensions(kv, (2, 0, 3, 1, 4))

        # k shape: B, self.num_heads, (self.dim // num_heads), self.Down_N
        # v shape: B, self.num_heads, self.Down_N, (self.dim // num_heads)
        k = keras_backend.permute_dimensions(kv[0], (0, 1, 3, 2))
        v = kv[1]
        # matmul shape: (B, self.num_heads, N, (self.dim // self.num_heads)) @ (B, self.num_heads, (self.dim // num_heads), self.Down_N)
        # attn.shape: 0B, self.num_heads, N, self.Down_N
        attn = tf.matmul(q, k) * self.scale
        attn = activations.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # matmul shape: (B, self.num_heads, N, self.Down_N) @ (B, self.num_heads, self.Down_N, (self.dim // num_heads)
        # out.shape: B, self.num_heads, N, (self.dim // num_heads)
        out = tf.matmul(attn, v)
        # shape: B, N, self.num_heads, (self.dim // num_heads)
        out = keras_backend.permute_dimensions(out, (0, 2, 1, 3))
        # shape: B, N, dim
        out = layers.Reshape((self.N, self.dim))(out)
        # shape: B, N, dim
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class EncodeBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=activations.gelu, norm_layer=custom_init_layer_norm,
                 sr_ratio=1, linear=False):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer()
        self.attn = SelfAttention(dim,
                                  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path, training=True) if drop_path > 0. else identity_layer
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EncoderMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop, linear=linear)

    def call(self, x, H, W):
        x_norm1 = self.norm1(x)
        x_attn1 = self.attn(x_norm1, H, W)
        x_drop_path1 = self.drop_path(x_attn1)

        x_norm2 = self.norm2(x)
        x_attn2 = self.mlp(x_norm2, H, W)
        x_drop_path2 = self.drop_path(x_attn2)
        out = x + x_drop_path1 + x_drop_path2

        return out


class DecodeBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=activations.gelu, norm_layer=custom_init_layer_norm,
                 sr_ratio=1, linear=False):
        super().__init__()
        self.dim = dim
        self.hidden_norm1 = norm_layer()
        self.current_norm1 = norm_layer()
        self.attn = Attention(dim,
                              num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path, training=True) if drop_path > 0. else identity_layer
        self.hidden_norm2 = norm_layer()
        self.current_norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DecoderMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop, linear=linear)

    def call(self, hidden, current, H, W):
        hidden_norm1 = self.hidden_norm1(hidden)
        current_norm1 = self.current_norm1(current)
        attn1 = self.attn(hidden_norm1, current_norm1, H, W)
        drop_path1 = self.drop_path(attn1)

        hidden_norm2 = self.hidden_norm1(hidden)
        current_norm2 = self.current_norm1(current)
        attn2 = self.mlp(hidden_norm2, current_norm2, H, W)
        drop_path2 = self.drop_path(attn2)
        out = current + drop_path1 + drop_path2

        return out


class OverlapPatchEmbed(layers.Layer):
    def __init__(self, patch_size=7, stride=4, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        self.stride = stride
        self.embed_dim = embed_dim
        self.patch_size = to2_tuple(patch_size)
        self.stride = stride
        self.proj = layers.Conv2D(filters=embed_dim,
                                  kernel_size=patch_size,
                                  strides=stride,
                                  padding="same",
                                  use_bias=True,
                                  kernel_initializer=dense_init,
                                  bias_initializer=dense_bias_init)
        self.positional_encode = AddPositionEmbs()
        self.norm = custom_init_layer_norm()

    def call(self, inputs):
        # input shape: B, H, W, ?
        # x shape: B, H, W, self.embed_dim
        x = self.proj(inputs)
        # shape: B, (H * W) // stride, self.embed_dim
        x = layers.Reshape((-1, self.embed_dim))(x)
        x = self.positional_encode(x)
        x = self.norm(x)

        return x


class DWConv(layers.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()

        self.dim = dim
        kernel_size = 3
        fan_out = (kernel_size ** 2) * dim
        fan_out //= dim
        normal_scale = math.sqrt(2.0 / fan_out)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                             strides=1,
                                             padding="same",
                                             use_bias=True,
                                             kernel_initializer=initializers.TruncatedNormal(
                                                 mean=0, stddev=normal_scale),
                                             bias_initializer=dense_bias_init)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class UnsharpMasking2D(layers.Layer):
#     def __init__(self, filters):
#         super(UnsharpMasking2D, self).__init__()
#         gauss_kernel_2d = get_gaussian_kernel(2, 0.0, 1.0)
#         self.gauss_kernel = tf.tile(
#             gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, filters, 1])

#         self.pointwise_filter = tf.eye(filters, batch_shape=[1, 1])

#     def call(self, input_tensor):
#         blur_tensor = tf.nn.separable_conv2d(input_tensor,
#                                              self.gauss_kernel,
#                                              self.pointwise_filter,
#                                              strides=[1, 1, 1, 1], padding='SAME')
#         unsharp_mask_tensor = 2 * input_tensor - blur_tensor
#         # because it used after tanh
#         unsharp_mask_tensor = tf.clip_by_value(unsharp_mask_tensor, -1, 1)
#         return unsharp_mask_tensor


# class HighwayMulti(layers.Layer):

#     activation = None
#     transform_gate_bias = None

#     def __init__(self, dim, activation='relu', transform_gate_bias=-3, **kwargs):
#         self.activation = activation
#         self.transform_gate_bias = transform_gate_bias
#         transform_gate_bias_initializer = Constant(self.transform_gate_bias)
#         self.dim = dim
#         self.dense_1 = Dense(
#             units=self.dim, bias_initializer=transform_gate_bias_initializer)

#         super(HighwayMulti, self).__init__(**kwargs)

#     def call(self, x, y):
#         transform_gate = self.dense_1(x)
#         transform_gate = layers.Activation("sigmoid")(transform_gate)
#         carry_gate = layers.Lambda(lambda x: 1.0 - x,
#                                    output_shape=(self.dim,))(transform_gate)
#         transformed_gated = layers.Multiply()([transform_gate, x])
#         identity_gated = layers.Multiply()([carry_gate, y])
#         value = Add()([transformed_gated, identity_gated])
#         return value

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = super(HighwayMulti, self).get_config()
#         config['activation'] = self.activation
#         config['transform_gate_bias'] = self.transform_gate_bias
#         return config


# class HighwayResnetBlock(layers.Layer):
#     def __init__(self, filters, use_highway=True):
#         super(HighwayResnetBlock, self).__init__()
#         # Define Base Model Params
#         self.use_highway = use_highway
#         self.depthwise_separable_conv = ConvBlock(
#             filters=filters, stride=1)
#         if self.use_highway is True:
#             self.highway_layer = HighwayMulti(dim=filters)

#     def call(self, input_tensor):

#         x = self.depthwise_separable_conv(input_tensor)
#         if self.use_highway is True:
#             x = self.highway_layer(x, input_tensor)
#         return x


# class HighwayResnetDecoder(layers.Layer):
#     def __init__(self, filters, unsharp=False):
#         super(HighwayResnetDecoder, self).__init__()
#         self.unsharp = unsharp
#         self.unsharp_mask_layer = UnsharpMasking2D(filters)

#         self.conv2d = HighwayResnetBlock(filters * 4, use_highway=False)
#         self.conv_after_pixel_shffle = HighwayResnetBlock(
#             filters, use_highway=False)

#         self.conv_before_upsample = HighwayResnetBlock(
#             filters, use_highway=False)
#         self.upsample_layer = layers.UpSampling2D(
#             size=2, interpolation="bilinear")
#         self.conv_after_upsample = HighwayResnetBlock(
#             filters, use_highway=False)

#         self.norm_layer = layers.BatchNormalization()
#         self.act_layer = tanh
#         self.highway_layer = HighwayMulti(dim=filters)

#     def call(self, input_tensor):

#         pixel_shuffle = self.conv2d(input_tensor)
#         pixel_shuffle = tf.nn.depth_to_space(pixel_shuffle, block_size=2)
#         pixel_shuffle = self.conv_after_pixel_shffle(pixel_shuffle)

#         x = self.conv_before_upsample(input_tensor)
#         x = self.upsample_layer(x)
#         x = self.conv_after_upsample(x)

#         output = self.highway_layer(pixel_shuffle, x)
#         output = self.norm_layer(output)
#         output = self.act_layer(output)
#         if self.unsharp:
#             output = self.unsharp_mask_layer(output)
#         return output
