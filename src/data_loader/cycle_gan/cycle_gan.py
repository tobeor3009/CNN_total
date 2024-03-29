# base module
from collections import deque
import random
# external module
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle
from tqdm import tqdm

# this library module
from ..utils import imread
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


class CycleGanDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 target_image_path_list,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_preprocess_input,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.target_image_path_dict = {index: target_image_path for index,
                                       target_image_path in enumerate(target_image_path_list)}
        self.data_len = len(self.image_path_dict)
        self.target_data_len = len(self.target_image_path_dict)
        self.data_on_ram_dict = {
            index: None for index in range(self.data_len)}
        self.target_data_on_ram_dict = {
            index: None for index in range(self.target_data_len)}

        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_image_channel = image_channel_dict["target_image"]
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.data_index_dict = {i: i for i in range(len(self))}
        self.target_data_index_dict = \
            {i: i for i in range(len(self.target_image_path_dict))}
        target_index_list = list(self.target_data_index_dict.values())
        random.shuffle(target_index_list)
        self.target_data_index_quque = deque(target_index_list)
        self.target_data_index_quque_len = self.target_data_len
        self.single_data_dict = \
            {"image_array": None, "target_image_array": None}
        if self.on_memory is True:
            self.augmentation_method = SegAugmentationPolicy(
                0, augmentation_policy_dict)
            self.image_preprocess_method = PreprocessPolicy(None)
            self.target_image_preprocess_method = PreprocessPolicy(
                None)
            self.get_unpaired_data_on_ram()

        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.target_image_preprocess_method = PreprocessPolicy(
            target_preprocess_input)
        self.augmentation_method = SegAugmentationPolicy(
            augmentation_proba, augmentation_policy_dict)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        if self.target_data_index_quque_len != 0:
            self.target_data_index_quque_len -= 1
        else:
            self.target_data_index_quque_len = self.target_data_len - 1
            target_index_list = list(self.target_data_index_dict.values())
            self.target_data_index_quque = deque(target_index_list)

        current_index = self.data_index_dict[i]
        target_index = self.target_data_index_quque.pop()

        if self.on_memory:
            image_array = self.data_on_ram_dict[current_index]
            target_image_array = self.target_data_on_ram_dict[target_index]

            image_array, target_image_array = self.augmentation_method(
                image_array, target_image_array)
            image_array = \
                self.image_preprocess_method(image_array)
            target_image_array = \
                self.target_image_preprocess_method(target_image_array)
        else:
            image_path = self.image_path_dict[current_index]
            target_image_path = self.target_image_path_dict[target_index]

            image_array = imread(image_path, channel=self.image_channel)
            target_image_array = imread(
                target_image_path, channel=self.target_image_channel)

            image_array = self.resize_method(image_array)
            target_image_array = self.resize_method(target_image_array)

            image_array, target_image_array = self.augmentation_method(
                image_array, target_image_array)

            image_array = \
                self.image_preprocess_method(image_array)
            target_image_array = \
                self.target_image_preprocess_method(target_image_array)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["target_image_array"] = target_image_array
        return self.single_data_dict

    def shuffle(self):
        data_index_list = syncron_shuffle(self.data_index_dict)
        target_data_index_list = syncron_shuffle(self.target_data_index_dict)
        for index, shuffled_index in enumerate(data_index_list):
            self.data_index_dict[index] = shuffled_index
        for index, shuffled_index in enumerate(target_data_index_list):
            self.target_data_index_dict[index] = shuffled_index

    def get_unpaired_data_on_ram(self):

        self.on_memory = False

        image_range = range(self.data_len)
        target_image_range = range(self.target_data_len)

        for index in tqdm(image_range):
            image_path = self.image_path_dict[index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = self.resize_method(image_array)

            self.data_on_ram_dict[index] = image_array

        for index in tqdm(target_image_range):
            target_image_path = self.target_image_path_dict[index]
            target_image_array = imread(
                target_image_path, channel=self.target_image_channel)
            target_image_array = self.resize_method(target_image_array)

            self.target_data_on_ram_dict[index] = target_image_array

        self.on_memory = True


class CycleGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 target_image_path_list=None,
                 batch_size=4,
                 on_memory=False,
                 augmentation_proba=None,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb", "target_image": "rgb"},
                 preprocess_input="-1~1",
                 target_preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = CycleGanDataGetter(image_path_list=image_path_list,
                                              target_image_path_list=target_image_path_list,
                                              on_memory=on_memory,
                                              augmentation_proba=augmentation_proba,
                                              augmentation_policy_dict=augmentation_policy_dict,
                                              image_channel_dict=image_channel_dict,
                                              preprocess_input=preprocess_input,
                                              target_preprocess_input=target_preprocess_input,
                                              target_size=target_size,
                                              interpolation=interpolation
                                              )
        self.batch_size = batch_size
        self.image_data_shape = self.data_getter[0]["image_array"].shape
        self.target_image_data_shape = self.data_getter[0]["target_image_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_target_image_array = np.zeros(
            (self.batch_size, *self.target_image_data_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_target_image_array[batch_index] = single_data_dict["target_image_array"]

        return self.batch_image_array, self.batch_target_image_array

    def print_data_info(self):
        data_num = self.data_getter.data_len
        target_data_num = self.data_getter.target_data_len
        print(
            f"Source data num {data_num}, Target data num: {target_data_num}")
