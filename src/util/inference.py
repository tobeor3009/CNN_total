import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from functools import partial
from .custom_loss import base_dice_loss
from tensorflow.keras import backend as keras_backend
from scipy.ndimage.measurements import label as scipy_get_label


def get_label(x):
    return scipy_get_label(x)[0]


def get_label_tf(x):
    return tf.py_function(func=get_label, inp=[x], Tout=tf.int32)


def remove_small_objects_single_channel(img, threshold, remove_region_area):
    binary = tf.where(img >= threshold, tf.ones_like(img), tf.zeros_like(img))
    binary = tf.cast(binary, tf.int32)

    labeled_image = get_label_tf(binary)
    labeled_image.set_shape(binary.shape)

    flatten_labeled_image = tf.reshape(labeled_image, [-1])
    unique_label, _, counts = tf.unique_with_counts(flatten_labeled_image)
    large_labels = tf.boolean_mask(unique_label, counts > remove_region_area)
    large_labels = tf.cast(large_labels, tf.float32)

    # Expand dimensions to make predictions 2D and then tile them to match the first dimension of the targets
    large_labels_expanded = tf.expand_dims(large_labels, 0)
    large_labels_tiled = tf.tile(large_labels_expanded,
                                 [tf.shape(flatten_labeled_image)[0], 1])
    filtered_binary = tf.math.in_top_k(targets=flatten_labeled_image,
                                       predictions=large_labels_tiled,
                                       k=tf.size(large_labels))
    filtered_binary = tf.reshape(filtered_binary, tf.shape(binary))
    return filtered_binary


def remove_small_objects(image_array, threshold=0.5, remove_region_ratio=0.01):
    mask_area = np.prod(image_array.shape[1:3])
    remove_region_area = mask_area * remove_region_ratio

    if image_array.shape[-1] == 1:
        region_mask = remove_small_objects_single_channel(image_array[..., 0], threshold,
                                                          remove_region_area)[..., tf.newaxis]
    else:
        filtered_binary_list = []
        for idx in range(image_array.shape[-1]):
            filtered_binary_list.append(remove_small_objects_single_channel(image_array[..., idx], threshold,
                                                                            remove_region_area)[..., tf.newaxis])
        region_mask = tf.concat(filtered_binary_list, axis=-1)
    region_mask = tf.cast(region_mask, tf.float32)
    return image_array * region_mask


def get_is_on_edge(mask_region_float):
    top_sum = tf.reduce_sum(mask_region_float[:, 0, :], axis=1)
    bottom_sum = tf.reduce_sum(mask_region_float[:, -1, :], axis=1)
    left_sum = tf.reduce_sum(mask_region_float[:, :, 0], axis=1)
    right_sum = tf.reduce_sum(mask_region_float[:, :, -1], axis=1)
    boundary_sum = top_sum + bottom_sum + left_sum + right_sum
    return tf.greater(boundary_sum, 0)


def remove_orange_peel_single_channel(img, threshold, remove_region_ratio=0.01):
    binary = tf.where(img >= threshold, tf.ones_like(img), tf.zeros_like(img))
    binary = tf.cast(binary, tf.int32)
    remove_region_area = img.shape[1] * img.shape[2] * remove_region_ratio

    labeled_image = get_label_tf(binary)
    labeled_image.set_shape(binary.shape)
    unique_label, _, counts = tf.unique_with_counts(
        tf.reshape(labeled_image, [-1]))

    mask = tf.ones_like(labeled_image, dtype=tf.bool)
    for label in (unique_label):
        mask_region = labeled_image == label
        mask_region_float = tf.cast(mask_region, dtype=tf.float32)
        is_on_edge = get_is_on_edge(mask_region_float)
        is_small_area = tf.reduce_sum(mask_region_float, axis=[
                                      1, 2]) < remove_region_area
        is_orange_peel = tf.logical_and(is_on_edge, is_small_area)
        remove_region = tf.logical_and(
            mask_region, is_orange_peel[:, None, None])
        mask = tf.where(remove_region, False, mask)
    return mask


def remove_orange_peel(image_array, threshold=0.5, remove_region_ratio=0.01):
    mask_area = np.prod(image_array.shape[1:3])
    remove_region_area = mask_area * remove_region_ratio

    if image_array.shape[-1] == 1:
        region_mask = remove_orange_peel_single_channel(image_array[..., 0], threshold,
                                                        remove_region_area)[..., tf.newaxis]
    else:
        filtered_binary_list = []
        for idx in range(image_array.shape[-1]):
            filtered_binary_list.append(remove_orange_peel_single_channel(image_array[..., idx], threshold,
                                                                          remove_region_area)[..., tf.newaxis])
        region_mask = tf.concat(filtered_binary_list, axis=-1)
    region_mask = tf.cast(region_mask, tf.float32)
    return image_array * region_mask


def get_best_row(df, key, mode):
    if mode == "max":
        best_value_index = df[key].idxmax()
    elif mode == "min":
        best_value_index = df[key].idxmin()
    best_value_row = df.loc[best_value_index]
    return best_value_row


def plot_best(csv_path, score_key, val_score_key, loss_key="loss", val_loss_key="val_loss", plot=True):
    df = pd.read_csv(csv_path)
    loss_min_row = get_best_row(df, loss_key, "min")
    score_max_row = get_best_row(df, score_key, "max")
    val_loss_min_row = get_best_row(df, val_loss_key, "min")
    val_score_max_row = get_best_row(df, val_score_key, "max")
    if plot:
        _, ax = plt.subplots(1, 2, figsize=(18, 6))

        ax[0].plot(df[loss_key])
        ax[0].plot(df[val_loss_key])
        ax[0].plot(loss_min_row["epoch"], loss_min_row[loss_key],
                   marker="|", color="black", markersize=30)
        ax[0].plot(val_loss_min_row["epoch"], val_loss_min_row[val_loss_key],
                   marker="|", color="red", markersize=30)
        ax[0].legend([loss_key, val_loss_key,
                      f"{loss_key}_min", f"{val_loss_key}_min"])

        ax[1].plot(df[score_key])
        ax[1].plot(df[val_score_key])
        ax[1].plot(score_max_row["epoch"], score_max_row[score_key],
                   marker="|", color="black", markersize=50)
        ax[1].plot(val_score_max_row["epoch"], val_score_max_row[val_score_key],
                   marker="|", color="red", markersize=30)
        ax[1].legend([score_key, val_score_key,
                      f"{score_key}_max", f"{val_score_key}_max"])
        plt.show()
        plt.close()
    return loss_min_row, score_max_row, val_loss_min_row, val_score_max_row


def find_epoch_from_folder(folder_path, epoch, pad_digit=2):
    real_epoch = epoch + 1
    find_key = '_{:0{}}.hdf5'.format(int(real_epoch), pad_digit)
    weight_path = [item for item in glob(
        f"{folder_path}/*.hdf5") if find_key in item]
    return weight_path[0]


def get_score(y_true, y_pred, threshold=0.5, axis=[1, 2, 3], remove_orange_peel=False):
    epsilon = keras_backend.epsilon()

    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)
    if remove_orange_peel:
        y_pred = remove_orange_peel(y_pred)
    tp = keras_backend.sum(y_true * y_pred, axis=axis)
    #tn = keras_backend.sum((1 - y_true) * (1 - y_pred), axis=axis)
    fp = keras_backend.sum(y_pred, axis=axis) - tp
    fn = keras_backend.sum(y_true, axis=axis) - tp
    dice_score = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    iou_score = (tp + epsilon) / (tp + fp + fn + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    return dice_score, iou_score, recall, precision


def dice_score(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)

    return 1 - base_dice_loss(y_true, y_pred)


def dice_score_remove_orange_peel(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)
    y_pred = remove_orange_peel(y_pred)
    return 1 - base_dice_loss(y_true, y_pred)


def get_iou(y_true, y_pred, threshold=0.5, axis=[1, 2, 3]):
    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)
    tp = keras_backend.sum(y_true * y_pred, axis=axis)
    tn = keras_backend.sum((1 - y_true) * (1 - y_pred), axis=axis)
    fp = keras_backend.sum(y_pred, axis=axis) - tp
    fn = keras_backend.sum(y_true, axis=axis) - tp
    return


def get_recall(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)
    recall, _ = tf.metrics.Recall()(y_true, y_pred)
    return recall


def get_precision(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true >= threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred >= threshold, dtype=tf.float32)
    precision, _ = tf.metrics.Precision()(y_true, y_pred)
    return precision
