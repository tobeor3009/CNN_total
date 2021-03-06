from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from segmentation_models.base import Loss

AXIS = [1, 2]
CHANNEL_WEIGHTED_AXIS = [1, 2]

PIE_VALUE = np.pi
SMOOTH = K.epsilon()


def get_channel_weighted_dice_loss(y_true, y_pred, beta=0.7, smooth=SMOOTH):
    alpha = 1 - beta

    y_true = (y_true + 1 + smooth) / 2
    y_pred = (y_pred + 1 + smooth) / 2

    prevalence_per_channel = K.mean(y_true, axis=CHANNEL_WEIGHTED_AXIS)

    # weight_per_channel = 1 / prevalence_per_channel
    # weight_per_channel_sum = K.sum(weight_per_channel, axis=-1)
    # weight_per_channel_sum = tf.expand_dims(weight_per_channel_sum, axis=-1)
    # weight_per_channel = weight_per_channel / weight_per_channel_sum

    tp = K.sum(y_true * y_pred, axis=CHANNEL_WEIGHTED_AXIS)
    fp = K.sum(y_pred, axis=CHANNEL_WEIGHTED_AXIS) - tp
    fn = K.sum(y_true, axis=CHANNEL_WEIGHTED_AXIS) - tp

    channel_weighted_dice_loss = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth) * \
        (smooth + prevalence_per_channel)

    channel_weighted_dice_loss = K.mean(channel_weighted_dice_loss)

    return -tf.math.log(channel_weighted_dice_loss)


def get_channel_weighted_mse_loss(y_true, y_pred, smooth=SMOOTH):

    prevalence_per_channel = \
        K.mean(y_true, axis=CHANNEL_WEIGHTED_AXIS)

    # weight_per_channel = 1 / prevalence_per_channel
    # weight_per_channel_sum = K.sum(weight_per_channel, axis=-1)
    # weight_per_channel_sum = K.expand_dims(weight_per_channel_sum, axis=-1)
    # weight_per_channel = weight_per_channel / weight_per_channel_sum

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


def rgb_color_histogram_loss(y_true, y_pred, element_num=256 * 256, value_range=(-1, 1), nbin=256):

    y_true_red = layers.Lambda(lambda x: x[:, 0, :, :])(y_true)
    y_true_green = layers.Lambda(lambda x: x[:, 1, :, :])(y_true)
    y_true_blue = layers.Lambda(lambda x: x[:, 2, :, :])(y_true)

    y_pred_red = layers.Lambda(lambda x: x[:, 0, :, :])(y_pred)
    y_pred_green = layers.Lambda(lambda x: x[:, 1, :, :])(y_pred)
    y_pred_blue = layers.Lambda(lambda x: x[:, 2, :, :])(y_pred)

    y_true_red_histo = tf.histogram_fixed_width(
        y_true_red, value_range, nbin=nbin)
    y_true_green_histo = tf.histogram_fixed_width(
        y_true_green, value_range, nbin=nbin)
    y_true_blue_histo = tf.histogram_fixed_width(
        y_true_blue, value_range, nbin=nbin)

    y_pred_red_histo = tf.histogram_fixed_width(
        y_pred_red, value_range, nbin=nbin)
    y_pred_green_histo = tf.histogram_fixed_width(
        y_pred_green, value_range, nbin=nbin)
    y_pred_blue_histo = tf.histogram_fixed_width(
        y_pred_blue, value_range, nbin=nbin)

    red_histo_loss = (y_true_red_histo - y_pred_red_histo) / element_num
    green_histo_loss = (y_true_green_histo -
                        y_pred_green_histo) / element_num
    blue_histo_loss = (y_true_blue_histo - y_pred_blue_histo) / element_num

    red_histo_loss = losses.mean_squared_error(red_histo_loss)
    green_histo_loss = losses.mean_squared_error(green_histo_loss)
    blue_histo_loss = losses.mean_squared_error(green_histo_loss)

    return red_histo_loss + green_histo_loss + blue_histo_loss
