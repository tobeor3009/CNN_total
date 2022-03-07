import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class PosEncodingLayer:
    def __init__(self, max_len, d_emb):
        self.pos_emb_matrix = layers.Embedding(max_len, d_emb, trainable=False,
                                               weights=[GetPosEncodingMatrix(max_len, d_emb)])

    def get_pos_seq(self, x):
        mask = backend.cast(backend.not_equal(x, 0), 'int32')
        pos = backend.cumsum(backend.ones_like(x, 'int32'), 1)
        return pos * mask

    def __call__(self, seq, pos_input=False):
        x = seq
        if not pos_input:
            x = layers.Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)


class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = backend.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = layers.Lambda(lambda x: x + pos)(x)
        return x


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, input_shape):
        super().__init__()
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, *input_shape)
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


class SelfAttention(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not heads == 1

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = layers.Softmax(axis=-1)
        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)
        self.to_out = Sequential(
            [
                layers.Dense(inner_dim, use_bias=False),
                layers.Dropout(dropout)
            ]
        ) if project_out else layers.Lambda(lambda x: x)

    def build(self, input_shape):
        super().build(input_shape)
        self.N = input_shape[1]

    def call(self, x):
        # qkv.shape : [B N 3 * dim_head]
        qkv = self.to_qkv(x)
        # qkv.shape : [B N 3 num_heads, dim_head]
        qkv = layers.Reshape((self.N, 3, self.heads, self.dim_head)
                             )(qkv)
        # shape: 3, B, self.num_heads, self.N, (dim_head)
        qkv = backend.permute_dimensions(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q.shape [B num_head N dim]
        # (k.T).shape [B num_head dim N]
        # dots.shape [B num_head N N]
        dots = tf.matmul(q, k, transpose_a=False, transpose_b=True)
        attn = self.attend(dots)
        # attn.shape [B num_head N N]
        # v.shape [B num_head N dim]
        # out.shape [B num_head N dim]
        out = tf.matmul(attn, v)
        # out.shape [B N (num_head * dim)]
        out = layers.Reshape((self.N, self.heads * self.dim_head))(out)
        out = self.to_out(out)
        return out


class Attention(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not heads == 1

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = layers.Softmax(axis=-1)
        self.to_q = layers.Dense(inner_dim, use_bias=False)
        self.to_kv = layers.Dense(inner_dim * 2, use_bias=False)
        self.to_out = Sequential(
            [
                layers.Dense(inner_dim, use_bias=False),
                layers.Dropout(dropout)
            ]
        ) if project_out else layers.Lambda(lambda x: x)

    def build(self, input_shape):
        super().build(input_shape)
        self.N = input_shape[1]

    def call(self, current, hidden):
        # qkv.shape : [B N 3 * dim_head]
        q = self.to_q(current)
        kv = self.to_kv(hidden)
        # qkv.shape : [B N 3 num_heads, dim_head]
        kv = layers.Reshape((self.N, 2, self.heads, self.dim_head)
                            )(kv)
        # shape: 3, B, self.num_heads, self.N, (dim_head)
        kv = backend.permute_dimensions(kv, (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]
        # q.shape [B num_head N dim]
        # (k.T).shape [B num_head dim N]
        # dots.shape [B num_head N N]
        dots = tf.matmul(q, k, transpose_a=False, transpose_b=True)
        attn = self.attend(dots)
        # attn.shape [B num_head N N]
        # v.shape [B num_head N dim]
        # out.shape [B num_head N dim]
        out = tf.matmul(attn, v)
        # out.shape [B N (num_head * dim)]
        out = layers.Reshape((self.N, self.heads * self.dim_head))(out)
        out = self.to_out(out)
        return out
