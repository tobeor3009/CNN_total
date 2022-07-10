from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from .base_model import InceptionResNetV2
from .base_model_as_class import InceptionResNetV2_progressive, Conv2DBN
from .layers import AddPositionEmbs, EqualizedConv, EqualizedDense, TransformerEncoder, get_act_layer
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


def get_inception_resnet_v2_transformer_disc(input_shape,
                                             padding="valid",
                                             base_act="relu",
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
        base_act=base_act,
        last_act=None,
        classifier_activation=None,
    )
    attn_dropout_proba = 0.1
    attn_dim_list = [block_size * 3 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)

    base_input = base_model.input
    # add a global spatial average pooling layer
    x = base_model.output
    x = base_model.output
    # (Batch_Size, 14, 14, 1536) => (Batch_Size, 28, 28, 384)
    x = tf.nn.depth_to_space(x, block_size=2)
    _, H, W, C = keras_backend.int_shape(x)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 384)
    x = layers.Reshape((H * W, C))(x)
    x = AddPositionEmbs(input_shape=(H * W, C))(x)
    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)
    predictions = layers.Dense(1, activation=last_act,
                               use_bias=False)(x)
    model = Model(base_input, predictions)
    return model


def get_inception_resnet_v2_discriminator(input_shape,
                                          padding="valid",
                                          base_act="relu",
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
        base_act=base_act,
        last_act=None,
        classifier_activation=None,
    )
    base_input = base_model.input

    # add a global spatial average pooling layer
    x = base_model.output
    predictions = layers.Activation(last_act)(x)
    model = Model(base_input, predictions)
    return model


def get_inception_resnet_v2_disc_stargan(input_shape,
                                         label_len,
                                         block_size=16,
                                         num_downsample=5,
                                         padding="valid",
                                         base_act="leakyrelu",
                                         last_act="sigmoid",
                                         ):
    target_shape = (input_shape[0] * (2 ** (5 - num_downsample)),
                    input_shape[1] * (2 ** (5 - num_downsample)),
                    input_shape[2])
    attn_dropout_proba = 0.1
    attn_dim_list = [block_size * 12 for _ in range(6)]
    num_head_list = [8 for _ in range(6)]
    attn_layer_list = []
    for attn_dim, num_head in zip(attn_dim_list, num_head_list):
        attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                        dropout=attn_dropout_proba)
        attn_layer_list.append(attn_layer)
    attn_sequence = Sequential(attn_layer_list)

    base_model = InceptionResNetV2_progressive(target_shape=target_shape,
                                               block_size=block_size,
                                               padding=padding,
                                               base_act=base_act,
                                               last_act=base_act,
                                               name_prefix="x2ct",
                                               num_downsample=num_downsample,
                                               use_attention=True)
    base_input = base_model.input
    base_output = base_model.output

    # (Batch_Size, 14, 14, 1536) => (Batch_Size, 28, 28, 384)
    _, H, W, C = keras_backend.int_shape(base_output)
    # (Batch_Size, 28, 28, 512) => (Batch_Size, 784, 384)
    x = layers.Reshape((H * W, C))(base_output)
    x = AddPositionEmbs(input_shape=(H * W, C))(x)

    x = attn_sequence(x)
    x = keras_backend.mean(x, axis=1)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)
    predictions = layers.Dense(label_len, activation='sigmoid',
                               use_bias=False)(x)
    validity = Conv2DBN(C, 3, activation=last_act)(base_output)

    model = Model(base_input, [validity, predictions])
    return model


def get_inception_resnet_v2_disc_wgan(input_shape,
                                      padding="valid",
                                      activation="relu",
                                      validity_act=None,
                                      block_size=16
                                      ):

    validity_model = InceptionResNetV2_progressive(target_shape=input_shape,
                                                   block_size=block_size,
                                                   padding=padding,
                                                   norm="instance",
                                                   base_act=activation,
                                                   last_act=activation,
                                                   name_prefix="validity",
                                                   num_downsample=5,
                                                   use_attention=True)

    input_tensor = layers.Input(input_shape)

    validity_encoded = validity_model(input_tensor)

    validity_pred = EqualizedConv(1, kernel=3)(validity_encoded)
    validity_pred = get_act_layer(validity_act)(validity_pred)
    model = Model(input_tensor, validity_pred)
    return model


def get_inception_resnet_v2_disc_stargan_wgan(input_shape,
                                              label_len,
                                              padding="valid",
                                              activation="relu",
                                              validity_act="sigmoid",
                                              block_size=16
                                              ):

    # attn_dropout_proba = 0.1
    # attn_dim_list = [block_size * 12 for _ in range(6)]
    # num_head_list = [8 for _ in range(6)]
    # attn_layer_list = []
    # for attn_dim, num_head in zip(attn_dim_list, num_head_list):
    #     attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
    #                                     dropout=attn_dropout_proba)
    #     attn_layer_list.append(attn_layer)
    # attn_sequence = Sequential(attn_layer_list)

    validity_model = InceptionResNetV2_progressive(target_shape=input_shape,
                                                   block_size=block_size,
                                                   padding=padding,
                                                   norm="instance",
                                                   base_act=activation,
                                                   last_act=activation,
                                                   name_prefix="validity",
                                                   num_downsample=5,
                                                   use_attention=True)

    class_model = InceptionResNetV2_progressive(target_shape=input_shape,
                                                block_size=block_size,
                                                padding=padding,
                                                norm="instance",
                                                base_act=activation,
                                                last_act=activation,
                                                name_prefix="class",
                                                num_downsample=5,
                                                use_attention=True)
    input_tensor = layers.Input(input_shape)

    validity_encoded = validity_model(input_tensor)
    class_encoded = class_model(input_tensor)
    _, H, W, C = class_encoded.shape
    class_encoded = layers.GlobalAveragePooling2D()(class_encoded)
    # class_encoded = layers.Reshape((H * W, C))(class_encoded)
    # class_encoded = AddPositionEmbs(input_shape=(H * W, C))(class_encoded)
    # class_encoded = attn_sequence(class_encoded)
    # class_encoded = keras_backend.mean(class_encoded, axis=1)

    validity_pred = EqualizedConv(1, kernel=3)(validity_encoded)
    validity_pred = get_act_layer(validity_act)(validity_pred)
    class_pred = EqualizedDense(C // 2)(class_encoded)
    class_pred = get_act_layer(activation)(class_pred)
    class_pred = layers.Dropout(0.2)(class_pred)
    class_pred = EqualizedDense(label_len)(class_pred)
    class_pred = get_act_layer("sigmoid")(class_pred)
    model = Model(input_tensor, [validity_pred, class_pred])
    return model


def get_inception_resnet_v2_disc_stargan_wgan_v2(input_shape,
                                                 label_len,
                                                 padding="valid",
                                                 norm="instance",
                                                 activation="relu",
                                                 validity_act="sigmoid",
                                                 block_size=16
                                                 ):

    validity_model = InceptionResNetV2_progressive(target_shape=input_shape,
                                                   block_size=block_size,
                                                   padding=padding,
                                                   norm=norm,
                                                   base_act=activation,
                                                   last_act=activation,
                                                   name_prefix="validity",
                                                   num_downsample=5,
                                                   use_attention=True)

    input_tensor = layers.Input(input_shape)

    encoded = validity_model(input_tensor)
    _, H, W, C = encoded.shape
    class_encoded = layers.GlobalAveragePooling2D()(encoded)

    validity_pred = EqualizedConv(1, kernel=3)(encoded)
    validity_pred = get_act_layer(validity_act)(validity_pred)
    class_pred = EqualizedDense(C // 2)(class_encoded)
    class_pred = get_act_layer(activation)(class_pred)
    class_pred = layers.Dropout(0.2)(class_pred)
    class_pred = EqualizedDense(label_len)(class_pred)
    class_pred = get_act_layer("sigmoid")(class_pred)
    model = Model(input_tensor, [validity_pred, class_pred])
    return model
