import os
import cv2
import json
import cupy as cp
import numpy as np
import tensorflow as tf
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
xyz_ref_white = np.array((0.950456, 1., 1.088754))


def Xray_to_L(image_array, use_gpu=False, uint8=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np

    arr_min, arr_max = image_array.min(), image_array.max()
    arr = image_array.astype(compat_cp.float32)
    arr = (arr - arr_min) / (arr_max - arr_min)

    mask = arr > 0.008856

    arr_converted = compat_cp.zeros_like(arr)
    arr_converted[mask] = compat_cp.cbrt(arr[mask])
    arr_converted[~mask] = 7.787 * arr[~mask] + (16 / 116)

    L = 116 * arr_converted - 16

    if uint8:
        L *= 2.55
        L = np.round(L).astype("uint8")
    else:
        L /= 100
        return L


def get_tissue_mask(I, luminosity_threshold=0.7, use_gpu=False):
    I_LAB = RGB_to_LAB(I, use_gpu=use_gpu)
    L = I_LAB[..., 0]
    # L = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)[:, :, 0].astype("float16")
    L = L / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
#     if mask.sum() == 0:
#         raise Exception("Empty tissue mask computed")

    return mask


def RGB_to_OD(I, use_gpu=False):

    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    I[I == 0] = 1
    OD = -1 * compat_cp.log(I / 255)
    if OD.ndim == 1:
        OD = OD[..., None]
    return OD


def OD_to_RGB(OD, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, 'Negative optical density'
    return (255 * compat_cp.exp(-1 * OD)).astype(np.uint8)


def soft_threshold(rho, lamda, use_gpu):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Soft threshold function used for normalized data and lasso regression'''
    lamda_below_index = rho < - lamda
    lamda_upper_index = rho > lamda

    new_rho = compat_cp.zeros_like(rho)
    new_rho[lamda_below_index] = rho[lamda_below_index] + lamda
    new_rho[lamda_upper_index] = rho[lamda_upper_index] - lamda

    return new_rho


def coordinate_descent_lasso(H, W, Y, lamda=0.1, num_iters=100, min_delta=0.9999, max_patience=10, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specifY whether or not we regularize H_0'''

    # Initialisation of useful values
    m, n = W.shape
    # normalizing W in case it was not done before
    W = W / (compat_cp.linalg.norm(W, axis=0))

    H_patience = 0
    H_previos_error = 1000
    H_current_error = compat_cp.linalg.norm(Y - W @ H)
    H_loss_decrease_ratio = H_current_error / H_previos_error

    # Looping until max number of iterations
    for iter_index in range(num_iters):
        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            W_j = W[:, j].reshape(-1, 1)
            Y_pred = W @ H
            rho = W_j.T @ (Y - Y_pred + H[j] * W_j)
            rho = rho.squeeze()
            H[j] = soft_threshold(rho, lamda, use_gpu=use_gpu)

        H_current_error = compat_cp.linalg.norm(Y - W @ H)
        H_loss_decrease_ratio = H_current_error / H_previos_error
        if H_loss_decrease_ratio > min_delta:
            H_patience += 1
            if H_patience > max_patience:
                break
        H_previos_error = H_current_error
    return H, iter_index


def coordinate_descent_lasso(H, W, Y, lamda=0.1, num_iters=100, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specifY whether or not we regularize H_0'''

    # Initialisation of useful values
    m, n = W.shape
    # normalizing W in case it was not done before
    W = W / (compat_cp.linalg.norm(W, axis=0))

    # Looping until max number of iterations
    for i in range(num_iters):
        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            W_j = W[:, j].reshape(-1, 1)
            Y_pred = W @ H
            rho = W_j.T @ (Y - Y_pred + H[j] * W_j)
            rho = rho.squeeze()
            H[j] = soft_threshold(rho, lamda, use_gpu=use_gpu)
    return H


def get_basis_concentraion_matrix(img_array, tissue_mask,
                                  input_channel_num, converted_channel_num,
                                  stop_ratio=0.9999, max_iter=200,
                                  h_lambda=0.1, use_gpu=False, verbose=0):
    if use_gpu:
        cp.cuda.runtime.setDevice(0)
        compat_cp = cp
    else:
        compat_cp = np

    img_array = img_array[tissue_mask]
    img_array = RGB_to_OD(img_array)
    img_array = compat_cp.array(img_array, dtype="float32")
    img_array = compat_cp.rollaxis(img_array, 1, 0)
    img_shape = img_array.shape
    elemenet_num = img_shape[-1]

    previous_error = 10000
    if use_gpu:
        w = compat_cp.random.random(size=(input_channel_num,
                                          converted_channel_num), dtype="float32")
        h = compat_cp.random.random(size=(converted_channel_num,
                                          elemenet_num), dtype="float32")
    else:
        w = compat_cp.random.random(size=(input_channel_num,
                                          converted_channel_num))
        h = compat_cp.random.random(size=(converted_channel_num,
                                          elemenet_num))
    for total_index in range(max_iter):

        for w_index in range(100):
            w = w * (img_array @ h.T) / (w @ h @ h.T + 1e-16)

        h = coordinate_descent_lasso(h, w, img_array,
                                     lamda=h_lambda, num_iters=100,
                                     use_gpu=use_gpu)

        for h2_index in range(5):
            h = h * (w.T @ img_array) / (w.T @ w @ h + 1e-16)

        current_error = compat_cp.linalg.norm(img_array - w @ h)
        if verbose:
            print(f"iter {total_index}: {current_error}")

        if stop_ratio is not None:
            if current_error / previous_error > stop_ratio:
                return previous_w, previous_h
        previous_w, previous_h = w, h
        previous_error = current_error

    # order H and E.
    # H on first row.
    if w[0, 0] < w[0, 1]:
        w = w[:, [1, 0]]

    return w, h


def get_concentraion_matrix(img_array, w, input_channel_num,
                            stop_ratio=0.9999, max_iter=200,
                            use_gpu=False, verbose=0):
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np

    img_array = RGB_to_OD(img_array).reshape(-1, input_channel_num)
    img_array = compat_cp.array(img_array, dtype="float32")
    img_array = compat_cp.rollaxis(img_array, 1, 0)
    img_shape = img_array.shape
    elemenet_num = img_shape[-1]

    converted_channel_num = w.shape[1]
    previos_error = 10000
    w = compat_cp.array(w)
    if use_gpu:
        h = compat_cp.random.random(
            size=(converted_channel_num, elemenet_num), dtype="float32")
    else:
        h = compat_cp.random.random(size=(converted_channel_num, elemenet_num))

    for total_index in range(max_iter):

        h = h * (w.T @ img_array) / (w.T @ w @ h + 1e-16)

        current_error = compat_cp.linalg.norm(img_array - w @ h)
        if verbose:
            print(f"stage2_iter {total_index}: {current_error}")

        if stop_ratio is not None:
            if current_error / previos_error > stop_ratio:
                break
        previos_error = current_error
    return h


def get_seperated_image(image_array, input_channel_num, converted_channel_num, use_gpu, stop_ratio=0.9999, verbose=0):

    tissue_mask = get_tissue_mask(image_array)
    source_w, tissue_h = get_basis_concentraion_matrix(image_array, tissue_mask,
                                                       input_channel_num, converted_channel_num,
                                                       stop_ratio=stop_ratio, max_iter=200,
                                                       h_lambda=1e-1, use_gpu=use_gpu,
                                                       verbose=verbose)
    image_h = get_concentraion_matrix(image_array, source_w, input_channel_num,
                                      stop_ratio=stop_ratio, max_iter=200,
                                      use_gpu=use_gpu, verbose=verbose)

    convert_image_list = []
    for idx in range(converted_channel_num):
        convert_image = source_w[:, idx:idx + 1] @ image_h[idx:idx + 1]
        convert_image = convert_image.transpose(
            1, 0).reshape(*image_array.shape)
        convert_image = OD_to_RGB(convert_image, use_gpu)
        if use_gpu:
            convert_image = convert_image.get()
        convert_image_list.append(convert_image)
    return convert_image_list
