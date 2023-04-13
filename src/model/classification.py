from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, Sequential, Model
from .inception_resnet_v2_unet_fix.layers import AddPositionEmbs, Pixelshuffle3D, get_act_layer
import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

DROPOUT_RATIO = 0.5


class SelfAttention(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

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
        qkv = keras_backend.permute_dimensions(qkv, (2, 0, 3, 1, 4))
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


class TransformerEncoder(layers.Layer):
    def __init__(self,
                 heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.attn = SelfAttention(heads, dim_head, dropout)
        self.attn_dropout = layers.Dropout(dropout)
        self.attn_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.ffpn_dense_1 = layers.Dense(inner_dim * 4, use_bias=False)
        self.ffpn_dense_2 = layers.Dense(inner_dim, use_bias=False)
        self.ffpn_dropout = layers.Dropout(dropout)
        self.ffpn_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, x):
        attn = self.attn(x)
        attn = self.attn_dropout(attn)
        attn = self.attn_norm(x + attn)

        out = self.ffpn_dense_1(attn)
        out = self.ffpn_dense_2(out)
        out = self.ffpn_dropout(out)
        out = self.ffpn_dropout(out)
        out = self.ffpn_norm(attn + out)

        return out


def get_inceptionv3_classification_model(input_shape, num_class,
                                         activation="binary_sigmoid",
                                         grad_cam=False,
                                         transfer_learning=False, train_mode="include_deep_layer",
                                         layer_name_frozen_to="mixed4",
                                         ):

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_model = InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        classes=None,
        pooling=None,
        classifier_activation=None
    )
    base_input = base_model.input

    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 14, 14, 2048)
    x = layers.GlobalAveragePooling2D()(x)
    # (Batch_Size, 2048)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(
            1, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(
            num_class, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(
            num_class, activation='softmax', dtype=dense_dtype)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model


def get_inceptionv3_classification_model_transformer(input_shape, num_class,
                                                     activation="binary_sigmoid",
                                                     feature_size=784,
                                                     attn_dim_list=[
                                                         64, 64, 64, 64, 64, 64],
                                                     num_head_list=[
                                                         8, 8, 8, 8, 8, 8],
                                                     grad_cam=False,
                                                     transfer_learning=False, train_mode="include_deep_layer",
                                                     layer_name_frozen_to="mixed4",
                                                     ):
    attn_dropout_proba = 0.3
    inner_dim = 2048
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)
    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")
    base_model = InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        classes=None,
        pooling=None,
        classifier_activation=None
    )
    base_input = base_model.input

    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 14, 14, 2048) => (Batch_Size, 28, 28, 512)
    x = tf.nn.depth_to_space(x, block_size=2)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 512)
    x = layers.Reshape((feature_size, 512))(x)
    x = AddPositionEmbs(input_shape=(feature_size, 512))(x)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(1, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(num_class, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(num_class, activation='softmax',
                                   dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model


def get_inception_resnet_v2_classification_model_transformer(input_shape, num_class,
                                                             activation="binary_sigmoid",
                                                             attn_dim_list=[
                                                                 48, 48, 48, 48, 48, 48],
                                                             num_head_list=[
                                                                 8, 8, 8, 8, 8, 8],
                                                             grad_cam=False,
                                                             transfer_learning=False, train_mode="include_deep_layer",
                                                             layer_name_frozen_to="mixed4",
                                                             ):
    attn_dropout_proba = 0.3
    inner_dim = 2048
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)
    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        classes=None,
        pooling=None,
        classifier_activation=None
    )
    base_input = base_model.input

    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 14, 14, 2048) => (Batch_Size, 28, 28, 512)
    x = tf.nn.depth_to_space(x, block_size=2)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 512)
    x = layers.Reshape((3600, 384))(x)
    x = AddPositionEmbs(input_shape=(3600, 384))(x)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(1, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(num_class, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(num_class, activation='softmax',
                                   dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model


def get_classification_model(base_model, num_class,
                             activation="binary_sigmoid",
                             grad_cam=False,
                             transfer_learning=False, train_mode="include_deep_layer",
                             layer_name_frozen_to="mixed4",
                             ):

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_input = base_model.input

    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 196, 2048)
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(
            1, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(
            num_class, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(
            num_class, activation='softmax', dtype=dense_dtype)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model


def get_label_inject_classification_model(base_model, inject_num_class, num_class,
                                          activation="binary_sigmoid",
                                          grad_cam=False,
                                          transfer_learning=False, train_mode="include_deep_layer",
                                          layer_name_frozen_to="mixed4",
                                          ):

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_input = base_model.input
    inject_label_input = layers.Input((inject_num_class,))
    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 196, 2048)
    x = layers.GlobalAveragePooling2D()(x)
    inject_label_channel = x.shape[-1] // 8
    inject_label = layers.Dense(inject_label_channel)(inject_label_input)
    x = layers.Concatenate(axis=-1)([x, inject_label])
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(
            1, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(
            num_class, activation='sigmoid', dtype=dense_dtype)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(
            num_class, activation='softmax', dtype=dense_dtype)(x)

    # this is the model we will train
    model = Model([base_input, inject_label_input], predictions)
    return model


def get_label_inject_classification_model_transformer(base_model, inject_num_class, num_class,
                                                      activation="binary_sigmoid",
                                                      feature_size=1024,
                                                      grad_cam=False,
                                                      transfer_learning=False, train_mode="include_deep_layer",
                                                      layer_name_frozen_to="mixed4",
                                                      ):
    attn_dropout_proba = 0.3

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_input = base_model.input
    inject_label_input = layers.Input((inject_num_class,))
    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break
    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 196, 2048)

    x = tf.nn.depth_to_space(x, block_size=2)
    _, H, W, C = x.shape
    inject_label_channel = C // 8
    x = layers.Reshape((H * W, C))(x)
    inject_label = layers.Dense(inject_label_channel)(inject_label_input)
    inject_label = layers.Reshape((1, inject_label_channel))(inject_label)
    inject_label = tf.tile(inject_label, (1, H * W, 1))
    x = layers.Concatenate(axis=-1)([x, inject_label])
    x = layers.Dense(feature_size)(x)
    x = get_act_layer("gelu")(x)
    x = AddPositionEmbs(input_shape=(H * W, feature_size))(x)

    attn_dim_list = [feature_size // 8 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(feature_size, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(1, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(num_class, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(num_class, activation='softmax',
                                   dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model([base_input, inject_label_input], predictions)

    return model


def get_label_inject_classification_model_transformer_3d(base_model, inject_num_class, num_class,
                                                         activation="binary_sigmoid",
                                                         grad_cam=False,
                                                         transfer_learning=False, train_mode="include_deep_layer",
                                                         layer_name_frozen_to="mixed4",
                                                         ):
    attn_dropout_proba = 0.3

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_input = base_model.input
    inject_label_input = layers.Input((inject_num_class,))
    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break
    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 196, 2048)
    _, Z, H, W, inject_label_channel = x.shape
    inject_label = layers.Dense(inject_label_channel)(inject_label_input)
    inject_label = layers.Reshape(
        (1, 1, 1, inject_label_channel))(inject_label)
    inject_label = tf.tile(inject_label, (1, Z, H, W, 1))
    x = layers.Concatenate(axis=-1)([x, inject_label])
    x = Pixelshuffle3D(2)(x)
    _, Z, H, W, C = x.shape
    x = layers.Reshape((Z * H * W, C))(x)
    x = AddPositionEmbs(input_shape=(Z * H * W, C))(x)

    attn_dim_list = [C // 8 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(1, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(num_class, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(num_class, activation='softmax',
                                   dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model([base_input, inject_label_input], predictions)
    return model


def get_classification_model_transformer_3d(base_model, num_class,
                                            activation="binary_sigmoid",
                                            grad_cam=False,
                                            transfer_learning=False, train_mode="include_deep_layer",
                                            layer_name_frozen_to="mixed4",
                                            ):
    attn_dropout_proba = 0.3

    if grad_cam is True:
        tf.keras.backend.set_floatx("float64")

    # create the base pre-trained model~
    base_input = base_model.input
    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break
    # add a global spatial average pooling layer
    x = base_model.output
    # (Batch_Size, 196, 2048)
    x = Pixelshuffle3D(2)(x)
    _, Z, H, W, C = x.shape
    x = layers.Reshape((Z * H * W, C))(x)
    x = AddPositionEmbs(input_shape=(Z * H * W, C))(x)

    attn_dim_list = [C // 8 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    if activation == "binary_sigmoid":
        predictions = layers.Dense(1, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_sigmoid":
        predictions = layers.Dense(num_class, activation='sigmoid',
                                   dtype=dense_dtype, use_bias=False)(x)
    elif activation == "categorical_softmax":
        predictions = layers.Dense(num_class, activation='softmax',
                                   dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model
