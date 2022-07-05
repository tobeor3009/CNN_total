import tensorflow as tf


def get_region(image_tensor, h, w, position):
    if position == "left_upper":
        image_tensor = image_tensor[:, :h, :w, :]
    elif position == "right_upper":
        image_tensor = image_tensor[:, :h, w:, :]
    elif position == "left_lower":
        image_tensor = image_tensor[:, h:, :w, :]
    elif position == "right_lower":
        image_tensor = image_tensor[:, h:, w:, :]
    return image_tensor


def recon_overlapping_patches(image_tensor):
    # _, H, W, _ = image_tensor_concat.shape
    _, H, W, _ = image_tensor[0].shape
    h, w = H // 2, W // 2
    # image_tensor = tf.split(image_tensor_concat, 10, -1)
    image_tensor_list = []
    row_0_col_0 = get_region(image_tensor[0], h, w, "left_upper") * 1
    row_0_col_1 = get_region(image_tensor[0], h, w, "right_upper") * \
        0.5 + get_region(image_tensor[1], h, w, "left_upper") * 0.5
    row_0_col_2 = get_region(image_tensor[1], h, w, "right_upper") * \
        0.5 + get_region(image_tensor[2], h, w, "left_upper") * 0.5
    row_0_col_3 = get_region(image_tensor[2], h, w, "right_upper") * 1

    row_1_col_0 = get_region(image_tensor[0], h, w, "left_lower") * \
        0.5 + get_region(image_tensor[3], h, w, "left_upper") * 0.5
    row_1_col_1 = get_region(image_tensor[0], h, w, "right_lower") * 0.25 + get_region(image_tensor[1], h, w, "left_lower") * 0.25 + \
        get_region(image_tensor[3], h, w, "right_upper") * 0.25 + \
        get_region(image_tensor[4], h, w, "left_upper") * 0.25
    row_1_col_2 = get_region(image_tensor[1], h, w, "right_lower") * 0.25 + get_region(image_tensor[2], h, w, "left_lower") * 0.25 + \
        get_region(image_tensor[4], h, w, "right_upper") * 0.25 + \
        get_region(image_tensor[5], h, w, "left_upper") * 0.25
    row_1_col_3 = get_region(image_tensor[2], h, w, "right_lower") * \
        0.5 + get_region(image_tensor[5], h, w, "right_upper") * 0.5

    row_2_col_0 = get_region(image_tensor[3], h, w, "left_lower") * \
        0.5 + get_region(image_tensor[6], h, w, "left_upper") * 0.5
    row_2_col_1 = get_region(image_tensor[3], h, w, "right_lower") * 0.25 + get_region(image_tensor[4], h, w, "left_lower") * 0.25 + \
        get_region(image_tensor[6], h, w, "right_upper") * 0.25 + \
        get_region(image_tensor[7], h, w, "left_upper") * 0.25
    row_2_col_2 = get_region(image_tensor[4], h, w, "right_lower") * 0.25 + get_region(image_tensor[5], h, w, "left_lower") * 0.25 + \
        get_region(image_tensor[7], h, w, "right_upper") * 0.25 + \
        get_region(image_tensor[8], h, w, "left_upper") * 0.25
    row_2_col_3 = get_region(image_tensor[5], h, w, "right_lower") * \
        0.5 + get_region(image_tensor[8], h, w, "right_upper") * 0.5

    row_3_col_0 = get_region(image_tensor[6], h, w, "left_lower") * 1
    row_3_col_1 = get_region(image_tensor[6], h, w, "right_lower") * \
        0.5 + get_region(image_tensor[7], h, w, "left_lower") * 0.5
    row_3_col_2 = get_region(image_tensor[7], h, w, "right_lower") * \
        0.5 + get_region(image_tensor[8], h, w, "left_lower") * 0.5
    row_3_col_3 = get_region(image_tensor[8], h, w, "right_lower") * 1

    image_tensor_list.append(row_0_col_0)
    image_tensor_list.append(row_0_col_1)
    image_tensor_list.append(row_0_col_2)
    image_tensor_list.append(row_0_col_3)
    image_tensor_list.append(row_1_col_0)
    image_tensor_list.append(row_1_col_1)
    image_tensor_list.append(row_1_col_2)
    image_tensor_list.append(row_1_col_3)
    image_tensor_list.append(row_2_col_0)
    image_tensor_list.append(row_2_col_1)
    image_tensor_list.append(row_2_col_2)
    image_tensor_list.append(row_2_col_3)
    image_tensor_list.append(row_3_col_0)
    image_tensor_list.append(row_3_col_1)
    image_tensor_list.append(row_3_col_2)
    image_tensor_list.append(row_3_col_3)

    row_1 = tf.concat(image_tensor_list[:4], axis=2)
    row_2 = tf.concat(image_tensor_list[4:8], axis=2)
    row_3 = tf.concat(image_tensor_list[8:12], axis=2)
    row_4 = tf.concat(image_tensor_list[12:], axis=2)
    restored = tf.concat([row_1, row_2, row_3, row_4], axis=1)

    return restored


# 1 / 4 Scale Restore
def recon_overlapping_patches_quarter_scale(image_tensor):
    # _, H, W, _ = image_tensor_concat.shape
    _, H, W, _ = image_tensor[0].shape
    h, w = H // 2, W // 2
    col_in_row = 7
    row_list = []

    col_list = []
    col_list.append(get_region(image_tensor[0], h, w, "left_upper") * 1)
    for idx in range(col_in_row - 1):
        col_element = get_region(image_tensor[idx], h, w, "right_upper") * \
            0.5 + get_region(image_tensor[idx + 1], h, w, "left_upper") * 0.5
        col_list.append(col_element)
    col_list.append(get_region(
        image_tensor[col_in_row - 1], h, w, "right_upper") * 1)
    row_list.append(col_list)

    for row_idx in range(col_in_row - 1):
        start_num = row_idx * col_in_row
        col_list = []
        col_list.append(get_region(image_tensor[start_num], h, w, "left_lower") *
                        0.5 + get_region(image_tensor[start_num + col_in_row], h, w, "left_upper") * 0.5)
        for col_idx in range(col_in_row - 1):
            col_element = get_region(image_tensor[start_num + col_idx], h, w, "right_lower") * 0.25 + \
                get_region(image_tensor[start_num + col_idx + 1], h, w, "left_lower") * 0.25 + \
                get_region(image_tensor[start_num + col_idx + col_in_row], h, w, "right_upper") * 0.25 + \
                get_region(image_tensor[start_num + col_idx + col_in_row + 1],
                           h, w, "left_upper") * 0.25
            col_list.append(col_element)
        col_list.append(get_region(image_tensor[start_num + col_in_row - 1], h, w, "right_lower") *
                        0.5 + get_region(image_tensor[start_num + col_in_row + col_in_row - 1], h, w, "right_upper") * 0.5)
        row_list.append(col_list)

    col_list = []
    start_num = (col_in_row - 1) * col_in_row
    col_list.append(get_region(
        image_tensor[start_num], h, w, "left_lower") * 1)
    for idx in range(col_in_row - 1):
        col_element = get_region(image_tensor[start_num + idx], h, w, "right_lower") * \
            0.5 + \
            get_region(image_tensor[start_num + idx + 1],
                       h, w, "left_lower") * 0.5
        col_list.append(col_element)
    col_list.append(get_region(
        image_tensor[start_num + col_in_row - 1], h, w, "right_lower") * 1)
    row_list.append(col_list)

    row_list = [tf.concat(row, axis=2) for row in row_list]
    restored = tf.concat(row_list, axis=1)

    return restored
