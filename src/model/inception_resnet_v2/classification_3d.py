from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from .base_model_as_class_3d import InceptionResNetV2_3D_Progressive
from .layers import AddPositionEmbs, EqualizedConv, EqualizedDense, TransformerEncoder, get_act_layer
np.random.seed(1337)  # for reproducibility

DROPOUT_RATIO = 0.5


def get_inception_resnet_v2_classification_model_transformer(input_shape, num_class,
                                                             padding="valid",
                                                             activation="sigmoid",
                                                             block_size=16,
                                                             z_downsample_list=[
                                                                 True, True, True, True, True],
                                                             grad_cam=False
                                                             ):
    attn_dropout_proba = 0.1
    attn_dim_list = [block_size * 12 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)

    if grad_cam:
        x *= 1e-1
        keras_backend.set_floatx('float64')
        dense_dtype = "float64"
    else:
        dense_dtype = "float32"

    base_model = InceptionResNetV2_3D_Progressive(
        target_shape=input_shape,
        block_size=block_size,
        padding=padding,
        groups=1,
        norm="batch",
        base_act="relu",
        last_act="relu",
        name_prefix="",
        num_downsample=5,
        z_downsample_list=z_downsample_list,
        use_attention=True,
        skip_connect_names=False
    )
    base_input = base_model.input

    # add a global spatial average pooling layer
    x = base_model.output
    _, Z, H, W, C = keras_backend.int_shape(x)
    x = layers.Reshape((Z * H * W, C))(x)
    x = AddPositionEmbs(input_shape=(Z * H * W, C))(x)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

    predictions = layers.Dense(num_class, activation=activation,
                               dtype=dense_dtype, use_bias=False)(x)

    # this is the model we will train
    model = Model(base_input, predictions)
    return model
