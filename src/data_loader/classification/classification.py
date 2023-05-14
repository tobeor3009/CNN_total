# base module
import math
from copy import deepcopy
# external module
import numpy as np

# this library module
from ..utils import imread, get_parent_dir_name, SingleProcessPool, MultiProcessPool, lazy_cycle
from ..base.base_loader import BaseDataGetter, BaseIterDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyAugmentationPolicy, \
    base_augmentation_policy_dict

def classification_collate_fn(data_object_list):
    batch_image_array = []
    batch_label_array = []
    for single_data_dict in data_object_list:
        image_array = single_data_dict["image_array"]
        label_array = single_data_dict["label"]
        batch_image_array.append(image_array)
        batch_label_array.append(label_array)
    batch_image_array = np.stack(batch_image_array, axis=0)
    batch_label_array = np.stack(batch_label_array, axis=0)

    return batch_image_array, batch_label_array


class ClassifyDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 label_level,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_dict,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.data_on_ram_dict = {}
        self.label_to_index_dict = label_to_index_dict
        self.label_level = label_level
        self.num_classes = len(self.label_to_index_dict)
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)
        self.categorize_method = \
            CategorizePolicy(class_mode, self.num_classes, dtype)

        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}

        self.augmentation_method = ClassifyAugmentationPolicy(
            0, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)
        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

        self.augmentation_method = \
            ClassifyAugmentationPolicy(
                augmentation_proba, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_dict["image"])

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
            image_array = self.resize_method(image_array)
            image_array = self.augmentation_method(image_array)
            image_array = self.preprocess_method(image_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
            else:
                image_dir_name = get_parent_dir_name(
                    image_path, self.label_level)
                label = self.label_to_index_dict[image_dir_name]
                label = self.categorize_method(label)
                self.class_dict[current_index] = label
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_class_cached = self.check_class_dict_cached()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["label"] = label

        return self.single_data_dict


class ClassifyDataloader(BaseIterDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 label_level=1,
                 batch_size=None,
                 num_workers=1,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_dict={"image": "-1~1"},
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
                                              label_to_index_dict=label_to_index_dict,
                                              label_level=label_level,
                                              on_memory=on_memory,
                                              augmentation_proba=augmentation_proba,
                                              augmentation_policy_dict=augmentation_policy_dict,
                                              image_channel_dict=image_channel_dict,
                                              preprocess_dict=preprocess_dict,
                                              target_size=target_size,
                                              interpolation=interpolation,
                                              class_mode=class_mode,
                                              dtype=dtype
                                              )
        self.data_num = len(self.data_getter)
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.shuffle = shuffle
        self.dtype = dtype
        if num_workers == 1:
            self.data_pool = SingleProcessPool(data_getter=self.data_getter,
                                               batch_size=self.batch_size,
                                               collate_fn=classification_collate_fn,
                                               shuffle=self.shuffle
                                               )
        else:
            self.data_pool = MultiProcessPool(data_getter=self.data_getter,
                                              batch_size=self.batch_size,
                                              num_workers=num_workers,
                                              collate_fn=classification_collate_fn,
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
        batch_label_array = []
        for total_index in range(start, end):
            single_data_dict = self.data_getter[total_index]

            batch_image_array.append(single_data_dict["image_array"])
            batch_label_array.append(single_data_dict["label"])
        batch_image_array = np.stack(batch_image_array, axis=0)
        batch_label_array = np.stack(batch_label_array, axis=0)

        return batch_image_array, batch_label_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
