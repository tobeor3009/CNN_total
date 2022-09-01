# base module
from copy import deepcopy
from tqdm import tqdm
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
                 image_path_list,
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

        self.argumentation_method = ClassifyArgumentationPolicy(
            0, argumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)
        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

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
            image_path = self.image_path_dict[current_index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = self.resize_method(image_array)
            image_array = self.argumentation_method(image_array)
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

    def get_data_on_disk(self):

        single_data_dict = self[0]
        image_array_shape = list(single_data_dict["image_array"].shape)
        image_array_shape = tuple([len(self)] + image_array_shape)
        image_array_dtype = single_data_dict["image_array"].dtype
        if self.class_mode == "categorical":
            label_array_shape = list(single_data_dict["label"].shape)
            label_array_shape = tuple([len(self)] + label_array_shape)
            label_array_dtype = single_data_dict["label"].dtype
        elif self.class_mode == "binary":
            label_array_shape = tuple([len(self)])
            label_array_dtype = np.int32

        # get_npy_array(path, target_size, data_key, shape, dtype)
        image_memmap_array, image_lock_path = get_npy_array(path=self.image_path_dict[0],
                                                            target_size=self.target_size,
                                                            data_key="image",
                                                            shape=image_array_shape,
                                                            dtype=image_array_dtype)
        label_memmap_array, label_lock_path = get_npy_array(path=self.image_path_dict[0],
                                                            target_size=self.target_size,
                                                            data_key="label",
                                                            shape=label_array_shape,
                                                            dtype=label_array_dtype)

        if os.path.exists(image_lock_path) and os.path.exists(label_lock_path):
            pass
        else:
            for index, single_data_dict in tqdm(enumerate(self)):
                image_array, label = single_data_dict.values()
                image_memmap_array[index] = image_array
                label_memmap_array[index] = label

            with open(image_lock_path, "w") as _, open(label_lock_path, "w") as _:
                pass
        array_dict_lazy = get_array_dict_lazy(key_tuple=("image_array", "label"),
                                              array_tuple=(image_memmap_array, label_memmap_array))
        self.data_on_ram_dict = LazyDict({
            i: (array_dict_lazy, i) for i in range(len(self))
        })
        self.on_memory = True


class ClassifyDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
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
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
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
