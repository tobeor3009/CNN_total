from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
import tensorflow as tf
import numpy as np
from .base_model import InceptionResNetV2
from .layers import AddPositionEmbs, TransformerEncoder
np.random.seed(1337)  # for reproducibility

DROPOUT_RATIO = 0.5


def get_inception_resnet_v2_classification_model_transformer(input_shape, num_class,
                                                             activation="binary_sigmoid",
                                                             attn_dim_list=[
                                                                 48, 48, 48, 48, 48, 48],
                                                             num_head_list=[
                                                                 8, 8, 8, 8, 8, 8],
                                                             grad_cam=False,
                                                             transfer_learning=False, train_mode="include_deep_layer",
                                                             layer_name_frozen_to="mixed4",
                                                             include_context=False
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
        padding='valid',
        classes=None,
        pooling=None,
        classifier_activation=None,
        include_context=include_context
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
    _, H, W, C = keras_backend.int_shape(x)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 512)
    x = layers.Reshape((H * W, C))(x)
    x = AddPositionEmbs(input_shape=(H * W, C))(x)
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
