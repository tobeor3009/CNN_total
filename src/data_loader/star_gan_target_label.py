# base module
import os
from copy import deepcopy
from collections import deque
from tqdm import tqdm
# external module
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyaugmentationPolicy, \
    base_augmentation_policy_dict


def get_diff_label(label_tensor):
    label_shape = np.shape(label_tensor)
    batch_size, num_class = label_shape[0], label_shape[1]
    label_origin = np.argmax(label_tensor, axis=1)
    label_tensor = np.random.rand(batch_size, num_class) - label_tensor
    for batch_idx in range(batch_size):
        label_idx = label_origin[batch_idx]
        label_tensor[batch_idx,
                     label_idx:] = label_tensor[batch_idx, label_idx:] * 2
    label_tensor = np.argmax(label_tensor, axis=1)
    label_tensor = np.eye(num_class)[label_tensor]
    return label_tensor


# def get_mmapdict(memmap_array_path, dict_key_list):
#     memmap_array = np.load(memmap_array_path)

#     for index, item in enumerate()

class StarGanDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_policy,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
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

        self.augmentation_method = ClassifyaugmentationPolicy(
            0, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)

        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

        self.augmentation_method = \
            ClassifyaugmentationPolicy(
                augmentation_proba, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array, label = \
                self.data_on_ram_dict[current_index].values()
            image_array = self.augmentation_method(image_array)

            image_array = self.preprocess_method(image_array)
        else:
            image_path = self.image_path_dict[current_index]

            image_array = imread(image_path, channel=self.image_channel)
            image_array, image_array_max, image_array_min = self.preprocess_method(
                image_array)
            image_array = self.resize_method(image_array)
            image_array = self.augmentation_method(image_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
            else:
                label = self.label_policy(image_path)
                self.class_dict[current_index] = label
                self.single_data_dict = deepcopy(self.single_data_dict)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["image_max"] = image_array_max
        self.single_data_dict["image_min"] = image_array_min
        self.single_data_dict["label"] = label
        return self.single_data_dict

    def get_data_on_ram(self):

        self.on_memory = False

        image_range = range(self.data_len)

        for index in tqdm(image_range):
            image_path = self.image_path_dict[index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = self.resize_method(image_array)

            label = self.label_policy(image_path)

            self.data_on_ram_dict[index] = \
                {"image_array": image_array, "label": label}

        self.on_memory = True


class StarGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_policy=None,
                 include_min_max=False,
                 batch_size=None,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = StarGanDataGetter(image_path_list=image_path_list,
                                             label_policy=label_policy,
                                             on_memory=on_memory,
                                             augmentation_proba=augmentation_proba,
                                             augmentation_policy_dict=augmentation_policy_dict,
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
        batch_diff_label_array = get_diff_label(self.batch_label_array)
        if self.include_min_max:
            return (self.batch_image_array, self.batch_label_array, batch_diff_label_array,
                    self.batch_image_max_array, self.batch_image_min_array)
        else:
            return self.batch_image_array, self.batch_label_array, batch_diff_label_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(
            f"Total data num {data_num} with {np.prod(self.label_data_shape)} classes")
