import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras import losses


histogram_loss_fn = losses.MeanAbsoluteError()


def rgb_color_histogram_loss(y_true, y_pred, element_num=256, value_range=(-1, 1), nbins=256):

    y_true_red = layers.Lambda(lambda x: x[:, :, :, 0])(y_true)
    y_true_green = layers.Lambda(lambda x: x[:, :, :, 1])(y_true)
    y_true_blue = layers.Lambda(lambda x: x[:, :, :, 2])(y_true)

    y_pred_red = layers.Lambda(lambda x: x[:, :, :, 0])(y_pred)
    y_pred_green = layers.Lambda(lambda x: x[:, :, :, 1])(y_pred)
    y_pred_blue = layers.Lambda(lambda x: x[:, :, :, 2])(y_pred)

    y_true_red_histo = tf.histogram_fixed_width(
        y_true_red, value_range, nbins=nbins)
    y_true_green_histo = tf.histogram_fixed_width(
        y_true_green, value_range, nbins=nbins)
    y_true_blue_histo = tf.histogram_fixed_width(
        y_true_blue, value_range, nbins=nbins)

    y_pred_red_histo = tf.histogram_fixed_width(
        y_pred_red, value_range, nbins=nbins)
    y_pred_green_histo = tf.histogram_fixed_width(
        y_pred_green, value_range, nbins=nbins)
    y_pred_blue_histo = tf.histogram_fixed_width(
        y_pred_blue, value_range, nbins=nbins)

    y_true_red_histo = keras_backend.cast(
        y_true_red_histo / element_num, 'float32')
    y_true_green_histo = keras_backend.cast(
        y_true_green_histo / element_num, 'float32')
    y_true_blue_histo = keras_backend.cast(
        y_true_blue_histo / element_num, 'float32')

    y_pred_red_histo = keras_backend.cast(
        y_pred_red_histo / element_num, 'float32')
    y_pred_green_histo = keras_backend.cast(
        y_pred_green_histo / element_num, 'float32')
    y_pred_blue_histo = keras_backend.cast(
        y_pred_blue_histo / element_num, 'float32')

    red_histo_loss = histogram_loss_fn(y_true_red_histo, y_pred_red_histo)
    green_histo_loss = histogram_loss_fn(
        y_true_green_histo, y_pred_green_histo)
    blue_histo_loss = histogram_loss_fn(y_true_blue_histo, y_pred_blue_histo)

    return red_histo_loss + green_histo_loss + blue_histo_loss
