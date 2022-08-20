# base module
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import os
# external module
import numpy as np
import progressbar
import random

# this library module
from .utils import imread, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, SegaugumentationPolicy, \
    base_augumentation_policy_dict


"""
Expected Data Path Structure

train - image
      - mask
valid - image
      - mask
test  - image
      - mask
"""

"""
To Be Done:
    - Test DataLoader Working
"""


class MultiScaleDataGetter(BaseDataGetter):
    def __init__(self,
                 image_folder_list,
                 on_memory,
                 augumentation_proba,
                 augumentation_policy_dict,
                 image_channel_dict,
                 preprocess_dict,
                 target_size,
                 interpolation,
                 use_multi_scale
                 ):
        super().__init__()

        self.folder_dict = {idx: image_folder
                            for idx, image_folder in enumerate(image_folder_list)}
        self.on_memory = on_memory

        self.image_channel_dict = image_channel_dict
        self.preprocess_dict = preprocess_dict
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = not on_memory
        self.data_index_dict = {idx: idx for idx in range(len(self))}
        self.single_data_dict = {"image_array": None,
                                 "mask_array": None,
                                 "label_array": None}
        self.image_name_list_1 = [[f"512_level_0_{idx}_image.png", f"512_level_0_{idx}_mask.png"] for idx in range(9)] + \
            [["512_level_1_image.png", "512_level_1_mask.png"]]
        self.image_name_list_2 = [[f"256_level_0_{idx}_image.png", f"256_level_0_{idx}_mask.png"] for idx in range(9)] + \
            [["256_level_1_image.png", "256_level_1_mask.png"]]

        if not use_multi_scale:
            self.image_name_list_1 = self.image_name_list_1[-1:]
            self.image_name_list_2 = self.image_name_list_2[-1:]
        self.augumentation_method = SegaugumentationPolicy(augumentation_proba,
                                                           augumentation_policy_dict)

    def getitem(self, idx, downscale):

        if idx >= len(self):
            raise IndexError

        current_index = self.data_index_dict[idx]
        if self.on_memory:
            image_array, mask_array, label_array = self.data_on_ram_dict[current_folder]
            image_array, mask_array = \
                self.augumentation_method(image_array, mask_array)
        else:
            folder = self.folder_dict[current_index]
            image_array_list = []
            mask_array_list = []
            image_name_list = self.image_name_list_2 if downscale else self.image_name_list_1
            for image_name, mask_name in image_name_list:
                image_array = imread(f"{folder}/{image_name}",
                                     channel=self.image_channel_dict["image"])
                mask_array = imread(f"{folder}/{mask_name}",
                                    channel=self.image_channel_dict["mask"])
                image_array_list.append(image_array)
                mask_array_list.append(mask_array)

            image_array = np.concatenate(image_array_list, axis=-1)
            mask_array = np.stack(mask_array_list, axis=-1)

            image_array = self.preprocess_dict["image"](image_array)
            mask_array = self.preprocess_dict["mask"](
                mask_array)
            # image_array = self.resize_method(image_array)
            # mask_array = self.resize_method(mask_array)
            image_array, mask_array = self.augumentation_method(image_array,
                                                                mask_array)
            if self.is_cached is False:
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_cached = None not in self.data_on_ram_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array

        return self.single_data_dict

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.folder_dict)

        return self.data_len

    def get_data_on_ram(self):
        self.on_memory = False

        for idx, single_data_dict in tqdm(enumerate(self)):
            image_array = single_data_dict["image_array"]
            mask_array = single_data_dict["mask_array"]
            self.data_on_ram_dict[idx] = (
                image_array, mask_array
            )
        self.on_memory = True


class MultiScaleDataloader(BaseDataLoader):

    def __init__(self,
                 image_folder_list=None,
                 batch_size=4,
                 on_memory=False,
                 augumentation_proba=None,
                 augumentation_policy_dict=base_augumentation_policy_dict,
                 image_channel_dict={"image": "rgb", "mask": "rgb"},
                 preprocess_dict=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32",
                 use_multi_scale=True):
        self.data_getter = MultiScaleDataGetter(image_folder_list=image_folder_list,
                                                on_memory=on_memory,
                                                augumentation_proba=augumentation_proba,
                                                augumentation_policy_dict=augumentation_policy_dict,
                                                image_channel_dict=image_channel_dict,
                                                preprocess_dict=preprocess_dict,
                                                target_size=target_size,
                                                interpolation=interpolation,
                                                use_multi_scale=use_multi_scale
                                                )
        self.batch_size = batch_size
        sample_data = self.data_getter.getitem(0, downscale=False)
        self.image_data_shape = sample_data["image_array"].shape
        self.mask_data_shape = sample_data["mask_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.use_multi_scale = use_multi_scale
        self.batch_image_array_1 = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_mask_array_1 = np.zeros(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)
        self.batch_image_array_2 = np.zeros(
            (self.batch_size * 4,
             *np.array(self.image_data_shape[:-1]) // 2,
             self.image_data_shape[-1]),
            dtype=self.dtype)
        self.batch_mask_array_2 = np.zeros(
            (self.batch_size * 4,
             *np.array(self.mask_data_shape[:-1]) // 2,
             self.mask_data_shape[-1]),
            dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        downscale = random.choice([True, False])
        if downscale:
            i = max(0, i // 4 - self.batch_size * 4)
            start = i * self.batch_size * 4
            end = start + self.batch_size * 4
            batch_image_array = self.batch_image_array_2
            batch_mask_array = self.batch_mask_array_2
        else:
            start = i * self.batch_size
            end = start + self.batch_size
            batch_image_array = self.batch_image_array_1
            batch_mask_array = self.batch_mask_array_1

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter.getitem(total_index, downscale)
            batch_image_array[batch_index] = single_data_dict["image_array"]
            batch_mask_array[batch_index] = single_data_dict["mask_array"]
        return batch_image_array, batch_mask_array[..., 5::6]

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
