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

    y_true = (y_true + 1 + smooth) / 2
    y_pred = (y_pred + 1 + smooth) / 2

    prevalence_per_channel = K.mean(y_true, axis=CHANNEL_WEIGHTED_AXIS)

    weight_per_channel = 1 / prevalence_per_channel
    weight_per_channel_sum = K.sum(weight_per_channel, axis=-1)
    weight_per_channel_sum = tf.expand_dims(weight_per_channel_sum, axis=-1)
    weight_per_channel = weight_per_channel / weight_per_channel_sum

    tp = K.sum(y_true * y_pred, axis=CHANNEL_WEIGHTED_AXIS)
    fp = K.sum(y_pred, axis=CHANNEL_WEIGHTED_AXIS) - tp
    fn = K.sum(y_true, axis=CHANNEL_WEIGHTED_AXIS) - tp

    channel_weighted_dice_loss = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth) * \
        (smooth + weight_per_channel)

    channel_weighted_dice_loss = K.mean(channel_weighted_dice_loss)

    return -tf.math.log(channel_weighted_dice_loss)


def get_channel_weighted_mse_loss(y_true, y_pred, smooth=SMOOTH):

    prevalence_per_channel = \
        K.mean(y_true, axis=CHANNEL_WEIGHTED_AXIS) + 1 + smooth
    prevalence_per_channel /= 2

    # weight_per_channel_sum = K.sum(prevalence_per_channel, axis=-1)
    # weight_per_channel_sum = tf.expand_dims(weight_per_channel_sum, axis=-1)
    # weight_per_channel = weight_per_channel_sum / prevalence_per_channel

    mse_loss = (y_true - y_pred) ** 2
    mse_loss = K.mean(mse_loss, axis=CHANNEL_WEIGHTED_AXIS)
    mse_loss = mse_loss * prevalence_per_channel

    channel_weighted_mse_loss = K.mean(mse_loss)

    return channel_weighted_mse_loss


def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):

    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss_1 = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_0 = - (1 - y_true) * ((1 - alpha) *
                               K.pow((y_pred), gamma) * K.log(1 - y_pred))
    loss = K.mean(loss_0 + loss_1)
    return loss
