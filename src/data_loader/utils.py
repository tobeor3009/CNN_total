import cv2
import os
import numpy as np


def imread(img_path, channel=None):
    img_byte_stream = open(img_path.encode("utf-8"), "rb")
    img_byte_array = bytearray(img_byte_stream.read())
    img_numpy_array = np.asarray(img_byte_array, dtype=np.uint8)
    img_numpy_array = cv2.imdecode(
        img_numpy_array, cv2.IMREAD_UNCHANGED)
    if channel == "rgb":
        img_numpy_array = cv2.cvtColor(
            img_numpy_array, cv2.COLOR_BGR2RGB)
    if len(img_numpy_array.shape) == 2:
        img_numpy_array = np.expand_dims(img_numpy_array, axis=-1)
    return img_numpy_array


def get_parent_dir_name(path):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)

    return abs_path.split(path_spliter)[-2]
