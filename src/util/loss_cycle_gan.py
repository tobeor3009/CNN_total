from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from . custom_loss_base import Loss

AXIS = [1, 2]
CHANNEL_WEIGHTED_AXIS = [1, 2]

PIE_VALUE = np.pi
SMOOTH = K.epsilon()


def get_channel_weighted_dice_loss(y_true, y_pred, beta=0.7, smooth=SMOOTH):
    alpha = 1 - beta
    prevalence_per_channel = \
        K.mean((y_true + 1) / 2, axis=CHANNEL_WEIGHTED_AXIS)

    tp = K.sum(y_true * y_pred, axis=CHANNEL_WEIGHTED_AXIS)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=CHANNEL_WEIGHTED_AXIS)
    fp = K.sum(y_pred, axis=CHANNEL_WEIGHTED_AXIS) - tp
    fn = K.sum(y_true, axis=CHANNEL_WEIGHTED_AXIS) - tp

    channel_weighted_dice_loss = \
        (tp + smooth) / (tp + alpha * fn + beta * fp + smooth) * \
        (smooth + prevalence_per_channel)

    channel_weighted_dice_loss = K.mean(channel_weighted_dice_loss)

    return -tf.math.log(channel_weighted_dice_loss)
