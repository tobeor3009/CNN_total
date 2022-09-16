from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow_addons.losses import WeightedKappaLoss
from tensorflow_addons.image import euclidean_dist_transform
from segmentation_models.base import Loss
from segmentation_models.losses import BinaryFocalLoss
from tensorflow.keras.losses import MeanAbsoluteError
mean_absolute_error = MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.AUTO)

AXIS = [1, 2]
PIE_VALUE = np.pi
SMOOTH = K.epsilon()
TVERSKY_BETA = 0.7
FOCAL_ALPHA = 0.25
FOCAL_BETA = 2.0
base_dice_loss = sm.losses.DiceLoss(per_image=True)


from scipy.ndimage import distance_transform_edt as distance
#####################################################################
# https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def boundary_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    multipled = tf.maximum(multipled, 0)
    return K.mean(multipled)
#####################################################################


def dice_loss(y_true, y_pred, per_image=False, smooth=SMOOTH):

    tp = K.sum(y_true * y_pred, axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    dice_score_per_image = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    dice_score_per_image = -1 * tf.math.log(dice_score_per_image)
    if per_image:
        return dice_score_per_image
    else:
        return K.mean(dice_score_per_image)


def dice_score(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    return 1 - base_dice_loss(y_true, y_pred)


def tversky_loss(y_true, y_pred, per_image=False, beta=0.7, smooth=SMOOTH):

    alpha = 1 - beta

    tp = K.sum(y_true * y_pred, axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp

    fn_and_fp = alpha * fn + beta * fp
    tversky_loss_per_image = (tp + smooth) / (tp + fn_and_fp + smooth)
    tversky_loss_per_image = -1 * tf.math.log(tversky_loss_per_image)
    if per_image:
        return tversky_loss_per_image
    else:
        return K.mean(tversky_loss_per_image)


def propotional_dice_loss(y_true, y_pred,
                          beta=0.7, per_image=False, smooth=SMOOTH,
                          channel_weight=None):

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
    total_score = negative_score + positive_score
    #total_score = -1 * tf.math.log(total_score)
    total_score = 1 - total_score
    if channel_weight is not None:
        channel_weight = np.array(channel_weight)
        channel_weight = K.constant(channel_weight)
        channel_weight_shape = (-1, 1, 1, channel_weight.shape[0])
        channel_weight = K.reshape(channel_weight, channel_weight_shape)
        total_score = total_score * channel_weight
    if per_image:
        return total_score
    else:
        return K.mean(total_score)


def cohens_kappa_loss(y_true, y_pred, per_image=False, smooth=SMOOTH):

    tp = K.sum(y_true * y_pred, axis=AXIS)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=AXIS)
    fp = K.sum(y_pred, axis=AXIS) - tp
    fn = K.sum(y_true, axis=AXIS) - tp
    numerator = (2 * (tp * tn - fn * fp))
    denominator = ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))

    score_per_image = tf.maximum(numerator / denominator, smooth)
    score_per_image = -1 * tf.math.log(score_per_image)

    if per_image:
        return score_per_image
    else:
        return K.mean(score_per_image)


def x2ct_loss(y_true, y_pred):

    mae_error = K.abs(y_true - y_pred)
    mae_error = K.mean(mae_error, axis=[1, 2, 3])
    #####################################################
    ################## Mean Projection ##################
    #####################################################
    gt_lat_projection = K.mean(y_true, axis=1)
    pred_lat_projection = K.mean(y_pred, axis=1)

    gt_ap_projection = K.mean(y_true, axis=2)
    pred_ap_projection = K.mean(y_pred, axis=2)

    gt_axial_projection = K.mean(y_true, axis=3)
    pred_axial_projection = K.mean(y_pred, axis=3)

    lat_projection_loss = K.abs(gt_lat_projection - pred_lat_projection)
    lat_projection_loss = K.mean(lat_projection_loss, axis=[1, 2])

    ap_projection_loss = K.abs(gt_ap_projection - pred_ap_projection)
    ap_projection_loss = K.mean(ap_projection_loss, axis=[1, 2])

    axial_projection_loss = K.abs(gt_axial_projection - pred_axial_projection)
    axial_projection_loss = K.mean(axial_projection_loss, axis=[1, 2])
    #####################################################
    ################## Max Projection ###################
    #####################################################
    gt_lat_max_projection = K.max(y_true, axis=1)
    pred_lat_max_projection = K.max(y_pred, axis=1)

    gt_ap_max_projection = K.max(y_true, axis=2)
    pred_ap_max_projection = K.max(y_pred, axis=2)

    gt_axial_max_projection = K.max(y_true, axis=3)
    pred_axial_max_projection = K.max(y_pred, axis=3)

    lat_projection_max_loss = K.abs(
        gt_lat_max_projection - pred_lat_max_projection)
    lat_projection_max_loss = K.mean(lat_projection_max_loss, axis=[1, 2])

    ap_projection_max_loss = K.abs(
        gt_ap_max_projection - pred_ap_max_projection)
    ap_projection_max_loss = K.mean(ap_projection_max_loss, axis=[1, 2])

    axial_projection_max_loss = K.abs(
        gt_axial_max_projection - pred_axial_max_projection)
    axial_projection_max_loss = K.mean(axial_projection_max_loss, axis=[1, 2])
    #####################################################
    #####################################################
    #####################################################

    projection_loss = (ap_projection_loss * 0.12 +
                       lat_projection_loss * 0.06 +
                       axial_projection_loss * 0.02)
    max_projection_loss = (ap_projection_max_loss * 0.06 +
                           lat_projection_max_loss * 0.03 +
                           axial_projection_max_loss * 0.01)

    return K.mean(mae_error + projection_loss + max_projection_loss)


class BoundaryLoss(Loss):
    def __init__(self):
        super().__init__(name='boundary_loss')
        self.loss_fn = \
            lambda y_true, y_pred: boundary_loss(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class ChannelWiseFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=4.0, channel_weight=None):
        self.base_focal_function = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.channel_weight_list = channel_weight

    def __call__(self, y_true, y_pred):
        loss = 0
        for index, channel_weight in enumerate(self.channel_weight_list):
            y_true_partial = y_true[:, :, :, index]
            y_pred_partial = y_pred[:, :, :, index]
            loss += self.base_focal_function(y_true_partial,
                                             y_pred_partial) * channel_weight
        return loss


class BaseDiceLoss(Loss):
    def __init__(self, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.loss_fn = lambda y_true, y_pred: \
            dice_loss(y_true, y_pred, per_image=per_image, smooth=smooth)

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class DiceLoss(Loss):
    def __init__(self, per_image=False, smooth=SMOOTH,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_BETA,
                 include_focal=False, include_boundary=False):
        super().__init__(name='tversky_loss')

        self.loss_fn = BaseDiceLoss(per_image=per_image, smooth=smooth)
        if include_focal is True:
            self.loss_fn += BinaryFocalLoss(alpha=alpha, gamma=gamma)
        if include_boundary is True:
            self.loss_fn += BoundaryLoss()

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class BaseTverskyLoss(Loss):
    def __init__(self, beta=TVERSKY_BETA, per_image=False, smooth=SMOOTH):
        super().__init__(name='tversky_loss')

        self.loss_fn = lambda y_true, y_pred: \
            tversky_loss(y_true, y_pred, beta=beta,
                         per_image=per_image, smooth=smooth)

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class TverskyLoss(Loss):
    def __init__(self, beta=TVERSKY_BETA, per_image=False, smooth=SMOOTH,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_BETA,
                 include_focal=False, include_boundary=False):
        super().__init__(name='tversky_loss')

        self.loss_fn = \
            BaseTverskyLoss(beta=beta, per_image=per_image, smooth=smooth)
        if include_focal is True:
            self.loss_fn += BinaryFocalLoss(alpha=alpha, gamma=gamma)
        if include_boundary is True:
            self.loss_fn += BoundaryLoss()

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class BasePropotionalDiceLoss(Loss):
    def __init__(self, beta=0.7, per_image=False, smooth=SMOOTH, channel_weight=None):
        super().__init__(name='propotional_dice_loss')

        self.loss_fn = \
            lambda y_true, y_pred: propotional_dice_loss(y_true,
                                                         y_pred,
                                                         beta=beta,
                                                         per_image=per_image,
                                                         smooth=smooth,
                                                         channel_weight=channel_weight)

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class PropotionalDiceLoss(Loss):
    def __init__(self, beta=TVERSKY_BETA, per_image=False, smooth=SMOOTH,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_BETA,
                 include_focal=False, include_boundary=False,
                 channel_weight=None):
        super().__init__(name='propotional_dice_loss')

        self.loss_fn = BasePropotionalDiceLoss(beta=beta,
                                               per_image=per_image,
                                               smooth=smooth,
                                               channel_weight=channel_weight)
        if include_focal is True:
            if channel_weight is not None:
                self.loss_fn = ChannelWiseFocalLoss(
                    alpha=alpha, gamma=gamma, channel_weight=channel_weight)
            else:
                self.loss_fn += BinaryFocalLoss(alpha=alpha, gamma=gamma)

        if include_boundary is True:
            self.loss_fn += BoundaryLoss()

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class BaseCohensKappaLoss(Loss):
    def __init__(self, per_image=False, smooth=SMOOTH):
        super().__init__(name='propotional_dice_loss')
        self.loss_fn = lambda y_true, y_pred: \
            cohens_kappa_loss(y_true, y_pred,
                              per_image=per_image, smooth=smooth)

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)


class CohensKappaLoss(Loss):
    def __init__(self, per_image=False, smooth=SMOOTH,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_BETA,
                 include_focal=False, include_boundary=False):
        super().__init__(name='cohens_kappa_loss')
        self.loss_fn = BaseCohensKappaLoss(per_image=per_image, smooth=smooth)
        if include_focal is True:
            self.loss_fn += BinaryFocalLoss(alpha=alpha, gamma=gamma)

        if include_boundary is True:
            self.loss_fn += BoundaryLoss()

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)
