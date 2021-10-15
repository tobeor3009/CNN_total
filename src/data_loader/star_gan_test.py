# base module
import os
from copy import deepcopy
from collections import deque
from tqdm import tqdm
# external module
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

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


# def get_mmapdict(memmap_array_path, dict_key_list):
#     memmap_array = np.load(memmap_array_path)

#     for index, item in enumerate()

class StarGanDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_policy,
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
        self.data_on_memory_dict = {}
        self.label_policy = label_policy
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.target_data_index_list = list(range(len(self)))
        self.target_data_index_list = \
            syncron_shuffle(self.target_data_index_list)
        self.target_data_index_quque_len = len(self)

        self.single_data_dict = {"image_array": None, "label": None}
        self.target_single_data_dict = {"image_array": None, "label": None}
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

        if self.target_data_index_quque_len != 0:
            self.target_data_index_quque_len -= 1
        else:
            self.target_data_index_quque_len = len(self) - 1
            self.target_data_index_list = syncron_shuffle(
                self.target_data_index_list)

        current_index = self.data_index_dict[i]
        target_index = self.target_data_index_list[i]

        if self.on_memory:
            image_array, label = \
                self.data_on_memory_dict[current_index].values()
            target_image_array, target_label = \
                self.data_on_memory_dict[target_index].values()
            image_array = self.argumentation_method(image_array)
            target_image_array = self.argumentation_method(target_image_array)

            image_array = self.preprocess_method(image_array)
            target_image_array = self.preprocess_method(target_image_array)
        else:
            image_path = self.image_path_dict[current_index]
            target_image_path = self.image_path_dict[target_index]

            image_array = imread(image_path, channel=self.image_channel)
            target_image_array = imread(
                target_image_path, channel=self.image_channel)

            image_array = self.resize_method(image_array)
            target_image_array = self.resize_method(target_image_array)

            image_array = self.argumentation_method(image_array)
            target_image_array = self.argumentation_method(target_image_array)

            image_array = self.preprocess_method(image_array)
            target_image_array = self.preprocess_method(target_image_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
                target_label = self.class_dict[target_index]
            else:
                label = self.label_policy(image_path)
                target_label = self.label_policy(target_image_path)

                self.class_dict[current_index] = label
                self.class_dict[target_index] = target_label

                self.single_data_dict = deepcopy(self.single_data_dict)
                self.target_single_data_dict = deepcopy(
                    self.target_single_data_dict)
                self.is_class_cached = self.check_class_dict_cached()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["label"] = label
        self.target_single_data_dict["image_array"] = target_image_array
        self.target_single_data_dict["label"] = target_label

        return self.single_data_dict, self.target_single_data_dict

    def get_data_on_ram(self):

        self.on_memory = False

        image_range = range(self.data_len)

        for index in tqdm(image_range):
            image_path = self.image_path_dict[index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = self.resize_method(image_array)

            label = self.label_policy(image_path)

            self.data_on_memory_dict[index] = \
                {"image_array": image_array, "label": label}

        self.on_memory = True

    def get_data_on_disk(self):

        single_data_dict, _ = self[0]
        image_array_shape = list(single_data_dict["image_array"].shape)
        image_array_shape = tuple([len(self)] + image_array_shape)
        image_array_dtype = single_data_dict["image_array"].dtype

        label_array_shape = list(single_data_dict["label"].shape)
        label_array_shape = tuple([len(self)] + label_array_shape)
        label_array_dtype = single_data_dict["label"].dtype

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
            image_range = range(self.data_len)

            for index in tqdm(image_range):
                image_path = self.image_path_dict[index]
                image_array = imread(image_path, channel=self.image_channel)
                image_array = self.resize_method(image_array)

                label = self.label_policy(image_path)

                image_memmap_array[index] = image_array
                label_memmap_array[index] = label

            with open(image_lock_path, "w") as _, open(label_lock_path, "w") as _:
                pass

        array_dict_lazy = get_array_dict_lazy(key_tuple=("image_array", "label"),
                                              array_tuple=(image_memmap_array, label_memmap_array))
        self.data_on_memory_dict = LazyDict({
            i: (array_dict_lazy, i) for i in range(len(self))
        })
        self.on_memory = True


class StarGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_policy=None,
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
        self.data_getter = StarGanDataGetter(image_path_list=image_path_list,
                                             label_policy=label_policy,
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
        temp_data = self.data_getter[0][0]
        self.image_data_shape = temp_data["image_array"].shape
        self.label_data_shape = temp_data["label"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_label_array = np.zeros(
            (self.batch_size, *self.label_data_shape), dtype=self.dtype)

        self.batch_target_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_target_label_array = np.zeros(
            (self.batch_size, *self.label_data_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict, target_single_data_dict = self.data_getter[total_index]

            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_label_array[batch_index] = single_data_dict["label"]

            self.batch_target_image_array[batch_index] = target_single_data_dict["image_array"]
            self.batch_target_label_array[batch_index] = target_single_data_dict["label"]

        return np.array([self.batch_image_array, self.batch_target_image_array]), \
            np.array([self.batch_label_array, self.batch_target_label_array])

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(
            f"Total data num {data_num} with {np.prod(self.label_data_shape)} classes")
