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


def restore_overlapping_patches(image_tensor_concat):
    _, H, W, _ = image_tensor_concat.shape
    h, w = H // 2, W // 2
    image_tensor = tf.split(image_tensor_concat, 10, -1)
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
