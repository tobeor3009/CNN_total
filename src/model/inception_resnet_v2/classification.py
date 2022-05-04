from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from .base_model import InceptionResNetV2
from .layers import AddPositionEmbs, TransformerEncoder
np.random.seed(1337)  # for reproducibility

DROPOUT_RATIO = 0.5


def get_inception_resnet_v2_classification_model_transformer(input_shape, num_class,
                                                             padding="valid",
                                                             activation="binary_sigmoid",
                                                             block_size=16,
                                                             grad_cam=False,
                                                             transfer_learning=False, train_mode="include_deep_layer",
                                                             layer_name_frozen_to="mixed4",
                                                             version="v1"
                                                             ):
    attn_dropout_proba = 0.1
    attn_dim_list = [block_size * 3 for _ in range(6)]
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

    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        block_size=block_size,
        padding=padding,
        classes=None,
        pooling=None,
        classifier_activation=None,
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
    # (Batch_Size, 14, 14, 1536) => (Batch_Size, 28, 28, 384)
    x = tf.nn.depth_to_space(x, block_size=2)
    _, H, W, C = keras_backend.int_shape(x)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 384)
    x = layers.Reshape((H * W, C))(x)
    if version == "v1":
        x = AddPositionEmbs(input_shape=(H * W, C))(x)
    elif version == "v2":
        x = layers.Permute((2, 1))(x)
        x = layers.Dense(32, activation=tf.nn.relu6,
                         dtype=dense_dtype, use_bias=False)(x)
        x = layers.Permute((2, 1))(x)
        # (Batch_Size, 32, 384)
        x = AddPositionEmbs(input_shape=(32, C))(x)

    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)

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


def get_inception_resnet_v2_discriminator(input_shape,
                                          padding="valid",
                                          activation="relu",
                                          last_act="sigmoid",
                                          block_size=16
                                          ):
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        block_size=block_size,
        padding=padding,
        classes=None,
        pooling=None,
        base_act=activation,
        last_act=None,
        classifier_activation=None,
    )
    base_input = base_model.input

    # add a global spatial average pooling layer
    x = base_model.output
    predictions = layers.Activation(last_act)(x)
    model = Model(base_input, predictions)
    return model


# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_discriminator(
    input_shape, filters=64,
    kernel_initializer=kernel_init,
    num_downsampling=5, name="disc"
):
    img_input = layers.Input(shape=input_shape, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model
