# base module
from copy import deepcopy
from tqdm import tqdm
from glob import glob
import os
# external module
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyArgumentationPolicy, \
    base_argumentation_policy_dict

"""
Expected Data Path Structure

Example)
train - negative
      - positive
valid - negative
      - positive
test - negative
     - positive

"""


class ClassifyDataGetter(BaseDataGetter):

    def __init__(self,
                 image_folder_list,
                 label_to_index_dict,
                 label_level,
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

        self.image_folder_dict = {idx: image_folder
                                  for idx, image_folder in enumerate(image_folder_list)}
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

        self.argumentation_method = ClassifyArgumentationPolicy(
            0, argumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)
        if self.on_memory is True:
            self.get_data_on_ram()

        self.argumentation_method = \
            ClassifyArgumentationPolicy(
                argumentation_proba, argumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array, label = \
                self.data_on_ram_dict[current_index].values()
            image_array = self.argumentation_method(image_array)
            image_array = self.preprocess_method(image_array)
        else:
            image_folder = self.image_folder_dict[current_index]
            image_path_list = sorted(glob(f"{image_folder}/*.png"))
            video_array = []
            for image_path in image_path_list:
                image_array = imread(image_path, channel=self.image_channel)
                image_array = self.resize_method(image_array)
                image_array = self.argumentation_method(image_array)
                video_array.append(image_array)
            video_array = np.stack(video_array, axis=0)
            video_array = self.preprocess_method(video_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
            else:
                image_dir_name = get_parent_dir_name(
                    image_folder, self.label_level)
                label = self.label_to_index_dict[image_dir_name]
                label = self.categorize_method(label)
                self.class_dict[current_index] = label
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_class_cached = self.check_class_dict_cached()

        self.single_data_dict["image_array"] = video_array
        self.single_data_dict["label"] = label

        return self.single_data_dict

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_folder_dict)

        return self.data_len


class ClassifyDataloader(BaseDataLoader):

    def __init__(self,
                 image_folder_list=None,
                 label_to_index_dict=None,
                 label_level=1,
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
        self.data_getter = ClassifyDataGetter(image_folder_list=image_folder_list,
                                              label_to_index_dict=label_to_index_dict,
                                              label_level=label_level,
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
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.image_data_shape = self.data_getter[0]["image_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_label_array = np.zeros(
            (self.batch_size, ), dtype=self.dtype)
        self.batch_label_array = self.data_getter.categorize_method(
            self.batch_label_array)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_label_array[batch_index] = single_data_dict["label"]

        return self.batch_image_array, self.batch_label_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
