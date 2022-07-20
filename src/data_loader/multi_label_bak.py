# base module
from copy import deepcopy
# external module
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, SegArgumentationPolicy, \
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
    - Test class cache
"""


class MultiLabelDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 label_to_index_dict,
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

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.mask_path_dict = {index: mask_path for index,
                               mask_path in enumerate(mask_path_list)}
        self.data_on_ram_dict = {}
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.categorize_method = CategorizePolicy(
            class_mode, self.num_classes, dtype)

        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}
        if self.on_memory is True:
            self.argumentation_method = SegArgumentationPolicy(
                0, argumentation_policy_dict)
            self.image_preprocess_method = PreprocessPolicy(None)
            self.mask_preprocess_method = PreprocessPolicy(None)
            self.get_data_on_ram()

        self.argumentation_method = SegArgumentationPolicy(
            argumentation_proba, argumentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy("mask")

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array, mask_array, label, preserve = self.data_on_ram_dict[current_index]
            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)
            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)
        else:
            image_path = self.image_path_dict[current_index]
            mask_path = self.mask_path_dict[current_index]

            image_array = imread(image_path, channel=self.image_channel)
            mask_array = imread(mask_path)
            image_array = self.resize_method(image_array)
            mask_array = self.resize_method(mask_array)

            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)

            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)

            if self.is_class_cached:
                label, preserve = self.class_dict[current_index]
            else:
                image_dir_name = get_parent_dir_name(image_path)
                label = self.label_to_index_dict[image_dir_name]
                label = self.categorize_method(label)
                preserve = np.mean(mask_array)

                self.class_dict[current_index] = label, preserve
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_class_cached = None not in self.class_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array
        self.single_data_dict["label"] = label
        self.single_data_dict["preserve"] = preserve

        return self.single_data_dict


class MultiLabelDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 on_memory=False,
                 argumentation_proba=False,
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32",
                 class_mode="binary"):
        self.data_getter = MultiLabelDataGetter(image_path_list=image_path_list,
                                                mask_path_list=mask_path_list,
                                                label_to_index_dict=label_to_index_dict,
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
        self.mask_data_shape = self.data_getter[0]["mask_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_mask_array = np.zeros(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)
        self.batch_label_array = np.zeros(
            (self.batch_size, ), dtype=self.dtype)
        self.batch_label_array = self.data_getter.categorize_method(
            self.batch_label_array)
        self.batch_preserve_array = np.zeros(
            (self.batch_size, ), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_mask_array[batch_index] = single_data_dict["mask_array"]
            self.batch_label_array[batch_index] = single_data_dict["label"]
            self.batch_preserve_array[batch_index] = single_data_dict["preserve"]
        return self.batch_image_array, (self.batch_mask_array, self.batch_label_array, self.batch_preserve_array)

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
