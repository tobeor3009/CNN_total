# base module
import os
import random
from copy import deepcopy
from collections import deque
from tqdm import tqdm
# external module
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyArgumentationPolicy, \
    base_argumentation_policy_dict

DATA_SLICE_NUM = 5


class StarGanDataGetter(BaseDataGetter):

    def __init__(self,
                 image_folder_list,
                 label_policy,
                 on_memory,
                 argumentation_proba,
                 argumentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_folder_dict = {idx: sorted(f"{image_folder}/*")
                                  for idx, image_folder in enumerate(image_folder_list)}
        self.data_on_ram_dict = {}
        self.label_policy = label_policy
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}

        self.single_data_dict = {"image_array": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}

        self.argumentation_method = ClassifyArgumentationPolicy(
            0, argumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)

        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

        self.argumentation_method = \
            ClassifyArgumentationPolicy(
                argumentation_proba, argumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            assert False, "half 3d dosen't support load on ram"
        else:
            image_path_list = self.image_folder_dict[current_index]
            slice_num = len(image_path_list)
            slice_idx = random.randint(slice_num - DATA_SLICE_NUM)
            slice_idx_list = range(slice_idx, slice_idx + DATA_SLICE_NUM)

            image_array_stacked = []
            for slice_idx in slice_idx_list:
                image_array = imread(
                    image_path_list[slice_idx], channel=self.image_channel)
                image_array, image_array_max, image_array_min = self.preprocess_method(
                    image_array)
                image_array = self.resize_method(image_array)
                image_array = self.argumentation_method(image_array)
                image_array_stacked.append(image_array)
            image_array_stacked = np.concatenate(image_array_stacked, axis=-1)
            if self.is_class_cached:
                label = self.class_dict[current_index]
            else:
                label = self.label_policy(image_path_list[0])
                self.class_dict[current_index] = label
                self.single_data_dict = deepcopy(self.single_data_dict)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["image_max"] = image_array_max
        self.single_data_dict["image_min"] = image_array_min
        self.single_data_dict["label"] = label

        return self.single_data_dict

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_folder_dict)

        return self.data_len


class StarGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_folder_list=None,
                 label_policy=None,
                 include_min_max=False,
                 batch_size=None,
                 on_memory=False,
                 argumentation_proba=False,
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = StarGanDataGetter(image_folder_list=image_folder_list,
                                             label_policy=label_policy,
                                             on_memory=on_memory,
                                             argumentation_proba=argumentation_proba,
                                             argumentation_policy_dict=argumentation_policy_dict,
                                             image_channel_dict=image_channel_dict,
                                             preprocess_input=preprocess_input,
                                             target_size=target_size,
                                             interpolation=interpolation,
                                             class_mode=class_mode,
                                             dtype=dtype
                                             )
        self.include_min_max = include_min_max
        self.batch_size = batch_size
        temp_data = self.data_getter[0]
        self.image_data_shape = temp_data["image_array"].shape
        self.label_data_shape = temp_data["label"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_image_max_array = np.zeros(
            (self.batch_size,), dtype=self.dtype)
        self.batch_image_min_array = np.zeros(
            (self.batch_size,), dtype=self.dtype)
        self.batch_label_array = np.zeros(
            (self.batch_size, *self.label_data_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]

            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_image_max_array[batch_index] = single_data_dict["image_max"]
            self.batch_image_min_array[batch_index] = single_data_dict["image_min"]
            self.batch_label_array[batch_index] = single_data_dict["label"]

        if self.include_min_max:
            return self.batch_image_array, self.batch_label_array, self.batch_image_max_array, self.batch_image_min_array
        else:
            return self.batch_image_array, self.batch_label_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(
            f"Total data num {data_num} with {np.prod(self.label_data_shape)} classes")
