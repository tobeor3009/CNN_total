# base module
import random
from glob import glob
# external module
import numpy as np
# this library module
from ..utils import imread, get_parent_dir_name
from ..base.base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, SegAugmentationPolicy, \
    base_augmentation_policy_dict

"""
Expected Data Path Structure

train - image
      - target_image
valid - image
      - target_image
test  - image
      - target_image
"""

"""
To Be Done:
    - Test DataLoader Working
"""

TARGET_RATIO_TERM = 0.02


class CycleGanDataGetter(BaseDataGetter):
    def __init__(self,
                 image_folder_list,
                 target_folder_list,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_preprocess_input,
                 target_size,
                 interpolation,
                 mode
                 ):
        super().__init__()

        self.mode = mode
        self.image_path_dict = {get_parent_dir_name(image_folder, level=0): glob(f"{image_folder}/*.npy")
                                for image_folder in image_folder_list}
        self.image_keys = list(self.image_path_dict.keys())
        self.target_image_path_dict = {get_parent_dir_name(image_folder, level=0): glob(f"{image_folder}/*.npy")
                                       for image_folder in target_folder_list}
        self.target_image_keys = list(self.target_image_path_dict.keys())
        self.__len__()

        if on_memory:
            print("level adjust cycle gan dataset not support on_memory.")
        else:
            self.on_memory = on_memory

        self.image_channel = image_channel_dict["image"]
        self.target_image_channel = image_channel_dict["target_image"]
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.single_data_dict = {
            "image_array": None, "target_image_array": None}

        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.target_image_preprocess_method = PreprocessPolicy(
            target_preprocess_input)
        self.augmentation_method = SegAugmentationPolicy(
            augmentation_proba, augmentation_policy_dict)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError
        # 0~1 random float
        random_float = random.uniform(0, 1)

        image_array = self.get_image_array(self.image_path_dict,
                                           self.image_keys, random_float,
                                           self.image_channel)
        random_float += random.uniform(-1 *
                                       TARGET_RATIO_TERM, TARGET_RATIO_TERM)
        random_float = np.clip(random_float, 0, 1)
        target_image_array = self.get_image_array(self.target_image_path_dict,
                                                  self.target_image_keys, random_float,
                                                  self.target_image_channel)

        image_array = self.resize_method(image_array)
        target_image_array = self.resize_method(target_image_array)

        image_array, image_max, image_min = \
            self.image_preprocess_method(image_array)
        target_image_array, target_image_max, target_image_min = \
            self.target_image_preprocess_method(target_image_array)

        image_array, target_image_array = self.augmentation_method(
            image_array, target_image_array)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["image_max"] = image_max
        self.single_data_dict["image_min"] = image_min
        self.single_data_dict["target_image_array"] = target_image_array
        self.single_data_dict["target_image_max"] = target_image_max
        self.single_data_dict["target_image_min"] = target_image_min
        return self.single_data_dict

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            self.data_len = 0
            for image_path_list in self.image_path_dict.values():
                self.data_len += len(image_path_list)
            self.target_data_len = 0
            for image_path_list in self.target_image_path_dict.values():
                self.target_data_len += len(image_path_list)
            return self.data_len

    def shuffle(self):
        pass

    def get_image_array(self, image_path_dict, image_keys, random_float, image_channel):
        key = random.choice(image_keys)
        image_path_list = image_path_dict[key]
        image_path_len = len(image_path_list)
        image_path_index = round(random_float * image_path_len) - 1
        if self.mode == "1d":
            image_path = image_path_list[image_path_index]
            image_array = imread(image_path, channel=image_channel)
            return image_array
        elif self.mode == "2.5d":
            image_path_index = np.clip(image_path_index, 1, image_path_len - 4)
            image_upper_1_path = image_path_list[image_path_index - 2]
            image_upper_2_path = image_path_list[image_path_index - 1]
            image_path = image_path_list[image_path_index]
            image_below_1_path = image_path_list[image_path_index + 1]
            image_below_2_path = image_path_list[image_path_index + 2]

            image_upper_1_array = imread(
                image_upper_1_path, channel=image_channel)
            image_upper_2_array = imread(
                image_upper_2_path, channel=image_channel)
            image_array = imread(image_path, channel=image_channel)
            image_below_1_array = imread(
                image_below_1_path, channel=image_channel)
            image_below_2_array = imread(
                image_below_2_path, channel=image_channel)
            return np.stack([image_upper_1_array, image_upper_2_array,
                            image_array,
                            image_below_1_array, image_below_2_array], axis=-1)

    def get_unpaired_data_on_ram(self):
        pass


class CycleGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_folder_list=None,
                 target_folder_list=None,
                 batch_size=4,
                 include_min_max=False,
                 on_memory=False,
                 augmentation_proba=None,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb", "target_image": "rgb"},
                 preprocess_input="-1~1",
                 target_preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 mode="1d",
                 dtype="float32"):
        self.data_getter = CycleGanDataGetter(image_folder_list=image_folder_list,
                                              target_folder_list=target_folder_list,
                                              on_memory=on_memory,
                                              augmentation_proba=augmentation_proba,
                                              augmentation_policy_dict=augmentation_policy_dict,
                                              image_channel_dict=image_channel_dict,
                                              preprocess_input=preprocess_input,
                                              target_preprocess_input=target_preprocess_input,
                                              target_size=target_size,
                                              interpolation=interpolation,
                                              mode=mode
                                              )
        self.batch_size = batch_size
        self.include_min_max = include_min_max
        temp_data = self.data_getter[0]
        self.image_data_shape = temp_data["image_array"].shape
        self.batch_image_max_shape = temp_data["image_max"].shape
        self.batch_image_min_shape = temp_data["image_min"].shape
        self.target_image_data_shape = temp_data["target_image_array"].shape
        self.target_image_max_shape = temp_data["target_image_max"].shape
        self.target_image_min_shape = temp_data["target_image_min"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_image_max = np.zeros(
            (self.batch_size, *self.batch_image_max_shape), dtype=self.dtype)
        self.batch_image_min = np.zeros(
            (self.batch_size, *self.batch_image_min_shape), dtype=self.dtype)
        self.batch_target_image_array = np.zeros(
            (self.batch_size, *self.target_image_data_shape), dtype=self.dtype)
        self.batch_target_image_max = np.zeros(
            (self.batch_size, *self.target_image_max_shape), dtype=self.dtype)
        self.batch_target_image_min = np.zeros(
            (self.batch_size, *self.target_image_min_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_image_max[batch_index] = single_data_dict["image_max"]
            self.batch_image_min[batch_index] = single_data_dict["image_min"]
            self.batch_target_image_array[batch_index] = single_data_dict["target_image_array"]
            self.batch_target_image_max[batch_index] = single_data_dict["target_image_max"]
            self.batch_target_image_min[batch_index] = single_data_dict["target_image_min"]

        if self.include_min_max:
            return ((self.batch_image_array, self.batch_target_image_array),
                    (self.batch_image_min, self.batch_image_max),
                    (self.batch_target_image_min, self.batch_target_image_max)
                    )
        else:
            return self.batch_image_array, self.batch_target_image_array

    def print_data_info(self):
        data_num = self.data_getter.data_len
        target_data_num = self.data_getter.target_data_len
        print(
            f"Source data num {data_num}, Target data num: {target_data_num}")
