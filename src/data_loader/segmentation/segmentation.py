# base module
from copy import deepcopy
import math
# external module
import numpy as np
from tensorflow.keras.utils import Sequence

# this library module
from ..utils import imread, SingleProcessPool, MultiProcessPool, lazy_cycle
from ..base.base_loader import BaseDataGetter, BaseIterDataLoader, \
    ResizePolicy, PreprocessPolicy, SegAugmentationPolicy, \
    base_augmentation_policy_dict


def seg_collate_fn(data_object_list):
    batch_image_array = []
    batch_mask_array = []
    for single_data_dict in data_object_list:
        image_array, mask_array = single_data_dict["image_array"], single_data_dict["mask_array"]
        batch_image_array.append(image_array)
        batch_mask_array.append(mask_array)
    batch_image_array = np.stack(batch_image_array, axis=0)
    batch_mask_array = np.stack(batch_mask_array, axis=0)

    return batch_image_array, batch_mask_array


class SegDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 imread_policy_dict,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 preprocess_dict,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.mask_path_dict = {index: mask_path for index,
                               mask_path in enumerate(mask_path_list)}
        self.data_on_ram_dict = {index: None for index, _
                                 in enumerate(image_path_list)}
        self.imread_policy_dict = imread_policy_dict
        self.on_memory = on_memory

        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "mask_array": None}
        self.augmentation_method = SegAugmentationPolicy(
            0, augmentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(None)
        self.mask_preprocess_method = PreprocessPolicy(None)
        if self.on_memory is True:
            self.get_data_on_ram()

        self.augmentation_method = SegAugmentationPolicy(augmentation_proba,
                                                         augmentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(
            preprocess_dict["image"])
        self.mask_preprocess_method = PreprocessPolicy(preprocess_dict["mask"])

        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array, mask_array = \
                self.data_on_ram_dict[current_index].values()
            image_array, mask_array = \
                self.augmentation_method(image_array, mask_array)
            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)
        else:
            image_path = self.image_path_dict[current_index]
            mask_path = self.mask_path_dict[current_index]

            image_array = imread(image_path, channel=None,
                                 policy=self.imread_policy_dict["image"])
            mask_array = imread(mask_path, channel=None,
                                policy=self.imread_policy_dict["mask"])
            image_array = self.resize_method(image_array)
            mask_array = self.resize_method(mask_array)

            image_array, mask_array = \
                self.augmentation_method(image_array, mask_array)

            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)

            if self.is_cached is False:
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_cached = None not in self.data_on_ram_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array
        return self.single_data_dict


class SegDataloader(BaseIterDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=None,
                 num_workers=1,
                 imread_policy_dict={"image": None, "mask": None},
                 on_memory=False,
                 augmentation_proba=None,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 preprocess_dict={"image": "-1~1", "mask": "mask"},
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         imread_policy_dict=imread_policy_dict,
                                         on_memory=on_memory,
                                         augmentation_proba=augmentation_proba,
                                         augmentation_policy_dict=augmentation_policy_dict,
                                         preprocess_dict=preprocess_dict,
                                         target_size=target_size,
                                         interpolation=interpolation
                                         )
        self.data_num = len(self.data_getter)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = dtype

        if num_workers == 1:
            self.data_pool = SingleProcessPool(data_getter=self.data_getter,
                                               batch_size=self.batch_size,
                                               collate_fn=seg_collate_fn,
                                               shuffle=self.shuffle
                                               )
        else:
            self.data_pool = MultiProcessPool(data_getter=self.data_getter,
                                              batch_size=self.batch_size,
                                              num_workers=num_workers,
                                              collate_fn=seg_collate_fn,
                                              shuffle=self.shuffle
                                              )
        self.print_data_info()
        self.on_epoch_end()

    def __iter__(self):
        return lazy_cycle(self.data_pool)

    def __next__(self):
        return next(self.data_pool)

    def __len__(self):
        return math.ceil(self.data_num / self.batch_size)

    def __getitem__(self, i):
        start = i * self.batch_size
        end = min(start + self.batch_size, self.data_num)

        batch_image_array = []
        batch_mask_array = []
        for total_index in range(start, end):
            single_data_dict = self.data_getter[total_index]

            batch_image_array.append(single_data_dict["image_array"])
            batch_mask_array.append(single_data_dict["mask_array"])
        batch_image_array = np.stack(batch_image_array, axis=0)
        batch_mask_array = np.stack(batch_mask_array, axis=0)

        return batch_image_array, batch_mask_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()


class SegDataSequence(Sequence):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=None,
                 imread_policy_dict={"image": None, "mask": None},
                 on_memory=False,
                 augmentation_proba=None,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 preprocess_dict={"image": "-1~1", "mask": "mask"},
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        super().__init__()
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         imread_policy_dict=imread_policy_dict,
                                         on_memory=on_memory,
                                         augmentation_proba=augmentation_proba,
                                         augmentation_policy_dict=augmentation_policy_dict,
                                         preprocess_dict=preprocess_dict,
                                         target_size=target_size,
                                         interpolation=interpolation
                                         )
        self.data_num = len(self.data_getter)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = dtype

        self.print_data_info()
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.data_num / self.batch_size)

    def __getitem__(self, i):
        start = i * self.batch_size
        end = min(start + self.batch_size, self.data_num)

        batch_image_array = []
        batch_mask_array = []
        for total_index in range(start, end):
            single_data_dict = self.data_getter[total_index]

            batch_image_array.append(single_data_dict["image_array"])
            batch_mask_array.append(single_data_dict["mask_array"])
        batch_image_array = np.stack(batch_image_array, axis=0)
        batch_mask_array = np.stack(batch_mask_array, axis=0)

        return batch_image_array, batch_mask_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()
