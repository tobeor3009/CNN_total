from .coord import CoordinateChannel2D
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
np.random.seed(1337)  # for reproducibility

DROPOUT_RATIO = 0.5


def get_inceptionv3_classification_model(input_shape, num_class,
                                         activation="binary_sigmoid",
                                         grad_cam=False,
                                         transfer_learning=False, train_mode="include_deep_layer",
                                         layer_name_frozen_to="mixed4",
                                         ):

    base_input = layers.Input(shape=(input_shape))
    coordinate_input = CoordinateChannel2D(use_radius=True)(base_input)
    # create the base pre-trained model~
    base_model = InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2] + 3),
        classes=None,
        pooling=None,
        classifier_activation=None
    )
    if transfer_learning:
        if train_mode == "dense_only":
            base_model.trainable = False
        elif train_mode == "include_deep_layer":
            for layer in base_model.layers:
                layer.trainable = False
                if layer.name == layer_name_frozen_to:
                    break

    # add a global spatial average pooling layer
    x = base_model(coordinate_input)
    # (Batch_Size,?)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATIO)(x)
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
