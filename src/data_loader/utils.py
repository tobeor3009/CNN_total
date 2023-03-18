import cv2
import os
import json
import math
import numpy as np
from collections.abc import Mapping
import SimpleITK as sitk
import nibabel as nib

def imread(img_path, channel=None, policy=None):
    extension = os.path.splitext(img_path)[1]
    
    if policy is not None:
        img_numpy_array = policy(img_path)
    elif extension == ".npy":
        img_numpy_array = np.load(
            img_path, allow_pickle=True).astype("float32")
    elif extension in [".gz", ".nii"]:
        image_object = nib.load(img_path)
        img_numpy_array = image_object.get_fdata().astype("float32")
    elif extension == ".dcm":
        image_slice = sitk.ReadImage(img_path)
        img_numpy_array = sitk.GetArrayFromImage(image_slice)[0, ...]
    else:
        img_byte_stream = open(img_path.encode("utf-8"), "rb")
        img_byte_array = bytearray(img_byte_stream.read())
        img_numpy_array = np.asarray(img_byte_array, dtype=np.uint8)
        if channel == "rgb":
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_UNCHANGED)
            img_numpy_array = cv2.cvtColor(
                img_numpy_array, cv2.COLOR_BGR2RGB)
        elif channel == "grayscale":
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_GRAYSCALE)
            img_numpy_array = np.expand_dims(img_numpy_array, axis=-1)
        else:
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_UNCHANGED)

    return img_numpy_array


def get_parent_dir_name(path, level=1):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)

    return abs_path.split(path_spliter)[-(1 + level)]


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def get_array_dict_lazy(key_tuple, array_tuple):
    return lambda index: {key: array[index] for key, array in zip(key_tuple, array_tuple)}


def get_npy_array(path, target_size, data_key, shape, dtype):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)
    data_sort_list = ["train", "valid", "test"]

    splited_path = abs_path.split(path_spliter)
    for index, folder in enumerate(splited_path):
        if folder == "datasets":
            break

    for folder in splited_path:
        find_data_sort = False

        for data_sort in data_sort_list:
            if folder == data_sort:
                find_data_sort = True
                break
        if find_data_sort is True:
            break
    # index mean datasets folder. so it means ~/datasets/task/name
    current_data_folder = path_spliter.join(splited_path[:index + 3])

    common_path = f"{current_data_folder}/{data_sort}_{target_size}_{data_key}"

    memmap_npy_path = f"{common_path}.npy"
    lock_path = f"{common_path}.lock"

    if os.path.exists(lock_path):
        memmap_array = np.memmap(
            memmap_npy_path, dtype=dtype, mode="r", shape=shape)
    else:
        memmap_array = \
            np.memmap(memmap_npy_path, dtype=dtype, mode="w+", shape=shape)

    return memmap_array, lock_path


def read_json_as_dict(json_path):
    json_file = open(json_path, encoding="utf-8")
    json_str = json_file.read()
    json_dict = json.loads(json_str)

    return json_dict


from time import sleep
import multiprocessing as mp
from multiprocessing.queues import Empty
import math
import random
import numpy as np

WAIT_TIME = 0.05


def lazy_cycle(iterable):
    while True:
        iter_init = iter(iterable)
        for element in iter_init:
            yield element


def default_collate_fn(data_object_list):

    batch_image_array = []
    batch_label_array = []
    for image_array, label_array in data_object_list:
        batch_image_array.append(image_array)
        batch_label_array.append(label_array)
    batch_image_array = np.stack(batch_image_array, axis=0)
    batch_label_array = np.stack(batch_label_array, axis=0)

    return batch_image_array, batch_label_array


def consumer_fn(data_getter, idx_queue, output_queue):
    inter_idx = 0
    while True:
        inter_idx += 1
        try:
            idx = idx_queue.get(timeout=0)
        except Empty:
            sleep(WAIT_TIME)
            continue
        if idx is None:
            break
        data = (idx, data_getter[idx])
        output_queue.put(data)


class BaseProcessPool():

    def __iter__(self):
        # print("iter called!")
        self.reset_states()
        return self

    def reset_states(self):
        self.batch_idx = 0
        self.batch_idx_list = None
        self.shuffle_idx()

    def shuffle_idx(self):
        if self.shuffle:
            random.shuffle(self.idx_list)


class SingleProcessPool(BaseProcessPool):
    def __init__(self, data_getter, batch_size,
                 collate_fn=default_collate_fn, shuffle=False):
        self.data_getter = data_getter
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.data_num = len(data_getter)
        self.idx_list = list(range(self.data_num))
        self.shuffle_idx()
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.batch_idx_list = None

    def __next__(self):
        if self.batch_idx == self.batch_num:
            self.reset_states()
            raise StopIteration
        start_idx = self.batch_size * self.batch_idx
        end_idx = min(start_idx + self.batch_size, self.data_num)
        batch_idx_list = self.idx_list[start_idx: end_idx]

        data_object_list = []
        for batch_idx in batch_idx_list:
            data_object = self.data_getter[batch_idx]
            data_object_list.append(data_object)
        batch_data_tuple = self.collate_fn(data_object_list)

        self.batch_idx_list = batch_idx_list
        self.batch_idx += 1
        return batch_data_tuple

    def __len__(self):
        return self.batch_num


class MultiProcessPool(BaseProcessPool):
    def __init__(self, data_getter, batch_size, num_workers,
                 collate_fn=default_collate_fn, shuffle=False):
        self.data_getter = data_getter
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.data_num = len(data_getter)
        self.idx_list = list(range(self.data_num))
        self.shuffle_idx()
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.batch_size = batch_size

        self.batch_idx = 0
        self.batch_idx_list = None

        self.idx_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.process_list = []
        for _ in range(num_workers):
            process = mp.Process(target=consumer_fn, args=(self.data_getter,
                                                           self.idx_queue,
                                                           self.output_queue))
            process.daemon = True
            process.start()
            self.process_list.append(process)

    def __next__(self):
        if self.batch_idx == self.batch_num:
            self.reset_states()
            raise StopIteration
        start_idx = self.batch_size * self.batch_idx
        end_idx = min(start_idx + self.batch_size, self.data_num)
        current_batch_size = end_idx - start_idx
        batch_idx_list = self.idx_list[start_idx: end_idx]
        for batch_idx in batch_idx_list:
            self.idx_queue.put(batch_idx)

        data_object_list = []
        for idx in range(current_batch_size):
            while True:
                try:
                    batch_idx, data_object = self.output_queue.get(timeout=0)
                    break
                except Empty:
                    sleep(WAIT_TIME)
                    continue
            data_object_list.append(data_object)

        batch_data_tuple = self.collate_fn(data_object_list)

        self.batch_idx_list = batch_idx_list
        self.batch_idx += 1
        return batch_data_tuple

    def __len__(self):
        return self.batch_num
