# base module
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import os
# external module
import numpy as np
import progressbar

# this library module
from .utils import imread, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, SegArgumentationPolicy, \
    base_argumentation_policy_dict


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
                 argumentation_proba,
                 argumentation_policy_dict,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.folder_dict = {idx: image_folder
                            for idx, image_folder in enumerate(image_folder_list)}
        self.data_on_ram_dict = {idx: None
                                 for idx, _ in enumerate(image_folder_list)}
        self.on_memory = on_memory

        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size // 4, interpolation)

        self.is_cached = not on_memory
        self.data_index_dict = {idx: idx for idx in range(len(self))}
        self.single_data_dict = {"image_array": None,
                                 "mask_array": None,
                                 "label_array": None}
        self.argumentation_method = SegArgumentationPolicy(
            0, argumentation_policy_dict)

        if self.on_memory is True:
            self.get_data_on_ram()

        self.argumentation_method = SegArgumentationPolicy(
            argumentation_proba, argumentation_policy_dict)

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        current_index = self.data_index_dict[idx]
        if self.on_memory:
            image_array, mask_array, label_array = self.data_on_ram_dict[current_folder]
            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)
        else:
            folder = self.folder_dict[current_index]
            image_array = np.load(
                f"{folder}/{self.target_size}_image.npy")
            mask_array = np.load(
                f"{folder}/{self.target_size}_mask.npy")
            label_array = np.load(
                f"{folder}/{self.target_size}_label.npy")
            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)
            image_array = (image_array / 127.5) - 1
            recon_array = self.resize_method(image_array)
            if self.is_cached is False:
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_cached = None not in self.data_on_ram_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["recon_array"] = recon_array
        self.single_data_dict["mask_array"] = mask_array
        self.single_data_dict["label_array"] = label_array

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
            label_array = single_data_dict["label_array"]
            self.data_on_ram_dict[idx] = (
                image_array, mask_array, label_array
            )
        self.on_memory = True


class MultiScaleDataloader(BaseDataLoader):

    def __init__(self,
                 image_folder_list=None,
                 batch_size=4,
                 on_memory=False,
                 argumentation_proba=None,
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = MultiScaleDataGetter(image_folder_list=image_folder_list,
                                                on_memory=on_memory,
                                                argumentation_proba=argumentation_proba,
                                                argumentation_policy_dict=argumentation_policy_dict,
                                                target_size=target_size,
                                                interpolation=interpolation
                                                )
        self.batch_size = batch_size
        sample_data = self.data_getter[0]
        self.image_data_shape = sample_data["image_array"].shape
        self.recon_data_shape = sample_data["recon_array"].shape
        self.mask_data_shape = sample_data["mask_array"].shape
        self.label_data_shape = sample_data["label_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_recon_array = np.zeros(
            (self.batch_size, *self.recon_data_shape), dtype=self.dtype)
        self.batch_mask_array = np.zeros(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)
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
            self.batch_recon_array[batch_index] = single_data_dict["recon_array"]
            self.batch_mask_array[batch_index] = single_data_dict["mask_array"]
            self.batch_label_array[batch_index] = single_data_dict["label_array"]
        return self.batch_image_array, self.batch_mask_array[..., 5::6]
        # return self.batch_image_array, (self.batch_recon_array, self.batch_mask_array, self.batch_label_array)

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
