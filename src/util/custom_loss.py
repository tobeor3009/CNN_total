from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow_addons.image import euclidean_dist_transform
from segmentation_models.base import Loss
from segmentation_models.losses import BinaryFocalLoss

AXIS = [1, 2, 3]
PIE_VALUE = np.pi
SMOOTH = K.epsilon()

dice_loss = sm.losses.DiceLoss(per_image=True)
binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
huber_loss = tf.keras.losses.Huber()
binary_focal_loss = BinaryFocalLoss(alpha=0.25, gamma=2.)


def calc_dist_map(mask_tensor):

    mask_tensor = K.round(mask_tensor)
    mask_tensor_uint8 = K.cast(mask_tensor, dtype="uint8")

    dist_map = euclidean_dist_transform(mask_tensor_uint8)
    dist_map = dist_map * mask_tensor

    return dist_map


def boundary_loss(y_true, y_pred):
    y_true_dist_map = calc_dist_map(y_true)
    y_pred_dist_map = calc_dist_map(y_pred)

    y_true_inside = y_true * (1 - y_pred)
    y_true_outside = (1 - y_true) * y_pred
    dist_map_diff = y_true_dist_map - y_pred_dist_map

    diff_inside = dist_map_diff * y_true_inside
    diff_outside = dist_map_diff * y_true_outside

    total_diff = diff_inside - diff_outside
    # total_diff = K.sum(total_diff) / K.sum(y_true)

    return K.mean(total_diff)


def dice_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    return 1 - dice_loss(y_true, y_pred)


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


def propotional_dice_loss(y_true, y_pred, beta=0.7, smooth=SMOOTH):

    alpha = 1 - beta
    prevalence = K.mean(y_true, axis=AXIS)

    tp = K.sum(y_true * y_pred, axis=AXIS)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    negative_score = (tn + smooth) \
        / (tn + beta * fn + alpha * fp + smooth) * (smooth + 1 - prevalence)
    positive_score = (tp + smooth) \
        / (tp + alpha * fn + beta * fp + smooth) * (smooth + prevalence)
    total_score = (negative_score + positive_score)
    total_score = -1 * tf.math.log(total_score)
    return K.mean(total_score)


class BoundaryLoss(Loss):
    def __init__(self):
        super().__init__(name='boundary_loss')
        self.loss_function = \
            lambda y_true, y_pred: boundary_loss(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)


class BaseTverskyLoss(Loss):
    def __init__(self, beta=0.7, per_image=False, smooth=SMOOTH):
        super().__init__(name='tversky_loss')

        self.loss_function = lambda y_true, y_pred: \
            tversky_loss(y_true, y_pred, beta=beta,
                         per_image=per_image, smooth=smooth)

    def __call__(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)


class TverskyLoss(Loss):
    def __init__(self, beta=0.7, per_image=False, smooth=SMOOTH,
                 alpha=0.25, gamma=2.0,
                 include_focal=False, include_boundary=False):
        super().__init__(name='tversky_loss')

        self.loss_function = \
            BaseTverskyLoss(beta=beta, per_image=per_image, smooth=smooth)
        if include_focal is True:
            self.loss_function += BinaryFocalLoss(alpha=alpha, gamma=gamma)
        if include_boundary is True:
            self.loss_function += BoundaryLoss()

    def __call__(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)


class BasePropotionalDiceLoss(Loss):
    def __init__(self, beta=0.7, smooth=SMOOTH):
        super().__init__(name='propotional_dice_loss')

        self.loss_function = \
            lambda y_true, y_pred: propotional_dice_loss(
                y_true, y_pred, beta=beta, smooth=smooth)

    def __call__(self, y_true, y_pred):

        return self.loss_function(y_true, y_pred)


class PropotionalDiceLoss(Loss):
    def __init__(self, beta=0.7, smooth=SMOOTH,
                 alpha=0.25, gamma=2.0,
                 include_focal=False, include_boundary=False):
        super().__init__(name='propotional_dice_loss')

        self.loss_function = \
            BasePropotionalDiceLoss(beta=beta, smooth=smooth)
        if include_focal is True:
            self.loss_function += BinaryFocalLoss(alpha=alpha, gamma=gamma)
        if include_boundary is True:
            self.loss_function += BoundaryLoss()

    def __call__(self, y_true, y_pred):

        return self.loss_function(y_true, y_pred)
