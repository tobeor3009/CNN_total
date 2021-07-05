from segmentation_models.base.functional import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from . custom_loss_base import Loss

AXIS = [1, 2, 3]
PIE_VALUE = np.pi
SMOOTH = K.epsilon()

dice_loss = sm.losses.DiceLoss(per_image=True)
binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
huber_loss = tf.keras.losses.Huber()
focal_loss = sm.losses.BinaryFocalLoss(alpha=0.25, gamma=4)

import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance


def combined_loss(y_true, y_pred):

    y_true = 1 - tf.math.cos(y_true * PIE_VALUE / 2)
    y_pred = 1 - tf.math.cos(y_pred * PIE_VALUE / 2)
    # y_true = 1 - tf.math.cos(y_true * PIE_VALUE / 2)
    # y_pred = 1 - tf.math.cos(y_pred * PIE_VALUE / 2)

    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred) + distribution_loss(y_true, y_pred)


def weighted_region_loss(y_true, y_pred, beta=0.7, smooth=SMOOTH):
    # y_pred = 1 - tf.math.cos(y_pred * PIE_VALUE / 2)
    tp = K.sum(y_true * y_pred, axis=AXIS)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    f1_loss_per_image = get_f1_loss_per_image(
        tp, tn, fp, fn, y_true, beta=beta, smooth=smooth)
    total_loss_per_image = f1_loss_per_image
    # accuracy_loss_per_image = get_accuracy_loss_per_image(tp, tn, fp, fn)
    # mse_loss = tf.math.reduce_mean((y_true - y_pred) ** 2, axis=AXIS)
    # total_loss_per_image = f1_loss_per_image + \
    #     accuracy_loss_per_image + mse_loss

    return K.mean(total_loss_per_image)


def get_f1_loss_per_image(tp, tn, fp, fn, y_true, beta=0.7, smooth=SMOOTH):
    alpha = 1 - beta
    prevalence = K.mean(y_true, axis=AXIS)

    negative_score = (tn + smooth) \
        / (tn + beta * fn + alpha * fp + smooth) * (smooth + 1 - prevalence)
    positive_score = (tp + smooth) \
        / (tp + alpha * fn + beta * fp + smooth) * (smooth + prevalence)
    total_score = (negative_score + positive_score)

    return -tf.math.log(total_score)


class WeightedRegionLoss(Loss):
    def __init__(self, beta=0.7, smooth=SMOOTH):
        super().__init__(name='weighted_region_loss')
        self.beta = beta
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return weighted_region_loss(y_true, y_pred, beta=self.beta, smooth=self.smooth)


def get_accuracy_loss_per_image(tp, tn, fp, fn):

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return 1 - accuracy


def f1_loss(y_true, y_pred, beta=1, smooth=K.epsilon()):

    tp = K.sum(y_true * y_pred, axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    return -tf.math.log(score)


def f1_loss(tp, fp, fn):

    score = (2 * tp + SMOOTH) \
        / (2 * tp + fn + fp + SMOOTH)

    return -tf.math.log(score)


def dice_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    return 1 - dice_loss(y_true, y_pred)


def get_tversky_loss(tp, fp, fn, alpha=0.3, beta=0.7, smooth=SMOOTH):

    fn_and_fp = alpha * fn + beta * fp
    total_score = (tp + smooth) / (tp + fn_and_fp + smooth)

    return 1 - total_score


def tversky_score(y_true, y_pred, per_image=False, beta=0.7, smooth=SMOOTH):

    alpha = 1 - beta

    tp = K.sum(y_true * y_pred, axis=AXIS)

    fp = K.sum(y_pred, axis=AXIS) - tp
    fp = K.sum((1 - y_true) * y_pred, axis=AXIS)

    fn = K.sum(y_true, axis=AXIS) - tp
    fn = K.sum(y_true * (1 - y_pred), axis=AXIS)

    fn_and_fp = alpha * fn + beta * fp
    tversky_score_per_image = (tp + smooth) / (tp + fn_and_fp)
    if per_image:
        return K.mean(tversky_score_per_image)
    else:
        return tversky_score_per_image


def tversky_loss(y_true, y_pred, per_image=False, beta=0.7, smooth=SMOOTH):

    alpha = 1 - beta

    tp = K.sum(y_true * y_pred, axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    fn_and_fp = alpha * fn + beta * fp
    tversky_loss_per_image = 1 - (tp + smooth) / (tp + fn_and_fp + smooth)

    if per_image:
        return K.mean(tversky_loss_per_image)
    else:
        return tversky_loss_per_image


class TverskyLoss(Loss):
    def __init__(self, beta=0.7, per_image=False, smooth=SMOOTH):
        super().__init__(name='tversky_loss')
        self.beta = beta
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return tversky_loss(y_true, y_pred, beta=self.beta, per_image=self.per_image, smooth=self.smooth)
