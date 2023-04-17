import tensorflow as tf

xyz_from_rgb = tf.constant([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]], dtype=tf.float32)
xyz_ref_white = tf.constant([0.950456, 1., 1.088754], dtype=tf.float32)


def RGB_to_LAB(image_array):
    arr = tf.reverse(image_array, axis=[-1])
    arr = arr / 255
    mask = arr > 0.04045
    arr = tf.where(mask, tf.pow((arr + 0.055) / 1.055, 2.4), arr / 12.92)
    arr = tf.matmul(arr, xyz_from_rgb, transpose_b=True)
    arr = arr / xyz_ref_white

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    mask = arr > 0.008856
    mask_x, mask_y, mask_z = mask[..., 0], mask[..., 1], mask[..., 2]

    arr_converted = tf.zeros_like(arr)
    arr_converted = tf.where(mask, tf.pow(arr, 1 / 3),
                             7.787 * arr + (16 / 116))

    x_converted, y_converted, z_converted = arr_converted[...,
                                                          0], arr_converted[..., 1], arr_converted[..., 2]

    L = tf.zeros_like(y)

    # Nonlinear distortion and linear transformation
    L = tf.where(mask_y, 116 * tf.pow(y, 1 / 3) - 16, 903.3 * y)
    L *= 2.55
    # if want to see this formula, go to https://docs.opencv.org/3.4.15/de/d25/imgproc_color_conversions.html RGB <-> CIELab
    a = 500 * (x_converted - y_converted) + 128
    b = 200 * (y_converted - z_converted) + 128

    return tf.stack([L, a, b], axis=-1)


def get_tissue_mask(I, luminosity_threshold=0.8):
    I_LAB = RGB_to_LAB(I)
    L = I_LAB[:, :, 0]
    L = L / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold
    return mask


def get_basis_concentraion_matrix(img_array, tissue_mask,
                                  input_channel_num, converted_channel_num,
                                  stop_ratio=0.9999, max_iter=200,
                                  h_lambda=0.1, use_gpu=False):

    img_array = img_array[tissue_mask]
    img_array = RGB_to_OD(img_array)
    img_array = np.array(img_array, dtype="float32")
    img_array = np.rollaxis(img_array, 1, 0)
    img_shape = img_array.shape
    elemenet_num = img_shape[-1]

    previous_error = 10000
    if use_gpu:
        w = np.random.random(size=(input_channel_num,
                                          converted_channel_num), dtype="float32")
        h = np.random.random(size=(converted_channel_num,
                                          elemenet_num), dtype="float32")
    else:
        w = np.random.random(size=(input_channel_num,
                                          converted_channel_num))
        h = np.random.random(size=(converted_channel_num,
                                          elemenet_num))
    for total_index in range(max_iter):

        for w_index in range(100):
            w = w * (img_array @ h.T) / (w @ h @ h.T + 1e-16)

        h = coordinate_descent_lasso(h, w, img_array,
                                     lamda=h_lambda, num_iters=100,
                                     use_gpu=use_gpu)

        for h2_index in range(5):
            h = h * (w.T @ img_array) / (w.T @ w @ h + 1e-16)

        current_error = np.linalg.norm(img_array - w @ h)
        print(f"iter {total_index}: {current_error}")

        if stop_ratio is not None:
            if current_error / previous_error > stop_ratio:
                return previous_w, previous_h
        previous_w, previous_h = w, h
        previous_error = current_error

    # order H and E.
    # H on first row.
#     if w[0, 0] < w[0, 1]:
#         w = w[:, [1, 0]]

    return w, h
