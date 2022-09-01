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
                 image_channel_dict,
                 preprocess_dict,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.image_folder_dict = {image_folder: sorted(glob(f"{image_folder}/level_*_image.png"), reverse=True)
                                  for image_folder in image_folder_list}
        self.mask_folder_dict = {image_folder: [image_path.replace("image", "mask") for image_path in image_path_list]
                                 for image_folder, image_path_list in self.image_folder_dict.items()}
        self.data_on_ram_dict = {image_folder: None
                                 for image_folder in self.image_folder_dict.keys()}
        self.on_memory = on_memory

        self.image_channel = image_channel_dict["image"]
        self.mask_channel = image_channel_dict["mask"]

        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = not on_memory
        self.data_index_dict = {idx: idx for idx in range(len(self))}
        self.data_folder_index_dict = {
            idx: image_folder for idx, image_folder in enumerate(self.image_folder_dict.keys())}
        self.single_data_dict = {"image_array": None,
                                 "mask_array": None,
                                 "label_array": None}
        self.argumentation_method = SegArgumentationPolicy(
            0, argumentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(
            preprocess_dict["image"])
        self.mask_preprocess_method = PreprocessPolicy(preprocess_dict["mask"])

        if self.on_memory is True:
            self.get_data_on_ram()

        self.argumentation_method = SegArgumentationPolicy(
            argumentation_proba, argumentation_policy_dict)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]
        current_folder = self.data_folder_index_dict[current_index]
        if self.on_memory:
            image_array, mask_array, class_label = self.data_on_ram_dict[current_folder]
            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)
        else:
            image_path_list = self.image_folder_dict[current_folder]
            mask_path_list = self.mask_folder_dict[current_folder]

            image_array_list = []
            mask_array_list = []
            for image_path, mask_path in zip(image_path_list, mask_path_list):
                image_array = imread(image_path, channel=self.image_channel)
                mask_array = imread(mask_path, channel=self.mask_channel)
                image_array_list.append(image_array)
                mask_array_list.append(mask_array)
            image_array = np.concatenate(image_array_list, axis=-1)
            mask_array = np.concatenate(mask_array_list, axis=-1)

            image_array = self.image_preprocess_method(image_array)
            mask_array, class_label = self.mask_preprocess_method(mask_array)

            image_array = self.resize_method(image_array).astype("float32")
            mask_array = self.resize_method(mask_array).astype("float32")

            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)

            if self.is_cached is False:
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_cached = None not in self.data_on_ram_dict.values()
                self.single_data_dict["image_folder"] = current_folder
        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array
        self.single_data_dict["label_array"] = class_label

        return self.single_data_dict

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_folder_dict)

        return self.data_len

    def get_data_on_ram(self):
        self.on_memory = False

        for index, single_data_dict in tqdm(enumerate(self)):
            image_array = single_data_dict["image_array"]
            mask_array = single_data_dict["mask_array"]
            label_array = single_data_dict["label_array"]
            current_folder = self.single_data_dict["image_folder"]
            self.data_on_ram_dict[current_folder] = (
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
                 image_channel_dict={"image": "rgb", "mask": None},
                 preprocess_dict={"image": "-1~1", "mask": "0~1"},
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = MultiScaleDataGetter(image_folder_list=image_folder_list,
                                                on_memory=on_memory,
                                                argumentation_proba=argumentation_proba,
                                                argumentation_policy_dict=argumentation_policy_dict,
                                                image_channel_dict=image_channel_dict,
                                                preprocess_dict=preprocess_dict,
                                                target_size=target_size,
                                                interpolation=interpolation
                                                )
        self.batch_size = batch_size
        self.image_data_shape = self.data_getter[0]["image_array"].shape
        self.mask_data_shape = self.data_getter[0]["mask_array"].shape
        self.label_data_shape = self.data_getter[0]["label_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
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
            self.batch_mask_array[batch_index] = single_data_dict["mask_array"]
            self.batch_label_array[batch_index] = single_data_dict["label_array"]

        return self.batch_image_array, (self.batch_image_array, self.batch_mask_array, self.batch_label_array)

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
