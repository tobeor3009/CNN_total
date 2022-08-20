# base module
import os
from copy import deepcopy
from collections import deque
from tqdm import tqdm
# external module
import numpy as np
import random
import albumentations as A
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyaugumentationPolicy, \
    base_augumentation_policy_dict


def split_multiclass_into_batch(real_img, label, img_min, img_max):
    b, h, w, c, num_class = real_img.shape
    target_label = np.random.normal(size=(b, num_class))
    target_label = np.argmax(target_label, axis=1)
    target_label = np.eye(num_class)[target_label]
    target_label_idx = np.argmax(target_label, axis=1)
    target_label = np.expand_dims(target_label, axis=1).astype("int32")
    target_label = np.repeat(target_label, num_class, axis=1)

    target_img = real_img[np.arange(b), ..., target_label_idx]
    target_img = np.expand_dims(target_img, axis=-1)
    target_img = np.repeat(target_img, num_class, axis=-1)
    target_img = np.transpose(target_img, (0, 4, 1, 2, 3))
    target_img = np.reshape(target_img, (b * num_class, h, w, c))

    real_img = np.transpose(real_img, (0, 4, 1, 2, 3))
    real_img = np.reshape(real_img, (b * num_class, h, w, c))

    label = np.reshape(label, (b * num_class, num_class))
    target_label = np.reshape(target_label, (b * num_class, num_class))
    img_min = np.reshape(img_min, (b * num_class,))
    img_max = np.reshape(img_max, (b * num_class,))

    return real_img, target_img, label, target_label, img_min, img_max


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


class StarGanDataGetter(BaseDataGetter):

    def __init__(self,
                 image_root_path,
                 label_list,
                 dicom_path_list,
                 label_policy,
                 apply_crop,
                 on_memory,
                 augumentation_proba,
                 augumentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_root_path = image_root_path
        self.label_list = label_list
        self.apply_crop = apply_crop
        self.one_hot_label = np.stack([label_policy(label) for label in label_list],
                                      axis=0)
        self.dicom_path_list = {idx: dicom_path for idx,
                                dicom_path in enumerate(dicom_path_list)}
        self.data_on_ram_dict = {}
        self.label_policy = label_policy
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.data_idx_dict = {i: i for i in range(len(self))}

        self.single_data_dict = {"image_array": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}

        self.augumentation_method = ClassifyaugumentationPolicy(
            0, augumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)

        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

        self.augumentation_method = \
            ClassifyaugumentationPolicy(
                augumentation_proba, augumentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_idx = self.data_idx_dict[i]

        dicom_path = self.dicom_path_list[current_idx]
        image_path_list = [f"{self.image_root_path}/{label}/{dicom_path}"
                           for label in self.label_list]

        image_array_list = []
        image_max_array = []
        image_min_array = []
        for image_path in image_path_list:
            image_array = imread(image_path, channel=self.image_channel)
            image_array, image_array_max, image_array_min = self.preprocess_method(
                image_array)
            image_array = self.resize_method(image_array)
            image_array = self.augumentation_method(image_array)
            image_array_list.append(image_array)
            image_max_array.append(image_array_max)
            image_min_array.append(image_array_min)
        if self.apply_crop:
            image_array_list = self.crop_method(image_array_list)
        image_array = np.stack(image_array_list, -1)
        image_max_array = np.array(image_max_array)
        image_min_array = np.array(image_min_array)

        self.single_data_dict = deepcopy(self.single_data_dict)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["image_max"] = image_max_array
        self.single_data_dict["image_min"] = image_min_array
        self.single_data_dict["label"] = self.one_hot_label

        return self.single_data_dict

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.dicom_path_list)
        return self.data_len

    def shuffle(self):
        data_index_list = syncron_shuffle(self.dicom_path_list)
        for index, shuffled_index in enumerate(data_index_list):
            self.dicom_path_list[index] = shuffled_index

    def crop_method(self, image_array_list):

        h_start = 0
        h_end = self.target_size[0] // 4 * 3
        h = random.randint(h_start, h_end)
        h_size = self.target_size[0] // 4

        w_start = 0
        w_end = self.target_size[1] // 4 * 3
        w = random.randint(w_start, w_end)
        w_size = self.target_size[1] // 4
        image_array_list = [image_array[h:h + h_size, w:w + w_size]
                            for image_array in image_array_list]
        return image_array_list


class StarGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_root_path=None,
                 label_list=None,
                 dicom_path_list=None,
                 label_policy=None,
                 include_min_max=False,
                 apply_crop=False,
                 batch_size=None,
                 on_memory=False,
                 augumentation_proba=False,
                 augumentation_policy_dict=base_augumentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = StarGanDataGetter(image_root_path=image_root_path,
                                             label_list=label_list,
                                             dicom_path_list=dicom_path_list,
                                             label_policy=label_policy,
                                             apply_crop=apply_crop,
                                             on_memory=on_memory,
                                             augumentation_proba=augumentation_proba,
                                             augumentation_policy_dict=augumentation_policy_dict,
                                             image_channel_dict=image_channel_dict,
                                             preprocess_input=preprocess_input,
                                             target_size=target_size,
                                             interpolation=interpolation,
                                             class_mode=class_mode,
                                             dtype=dtype
                                             )
        self.include_min_max = include_min_max
        self.label_num = len(label_list)
        self.batch_size = batch_size
        temp_data = self.data_getter[0]
        image_data_shape = temp_data["image_array"].shape
        image_max_shape = temp_data["image_max"].shape
        image_min_shape = temp_data["image_min"].shape
        label_data_shape = temp_data["label"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.batch_image_array = np.zeros((self.batch_size, *image_data_shape),
                                          dtype=self.dtype)
        self.batch_image_max_array = np.zeros((self.batch_size, *image_max_shape),
                                              dtype=self.dtype)
        self.batch_image_min_array = np.zeros((self.batch_size, *image_min_shape),
                                              dtype=self.dtype)
        self.batch_label_array = np.zeros((self.batch_size, *label_data_shape),
                                          dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_idx, total_idx in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_idx]
            self.batch_image_array[batch_idx] = single_data_dict["image_array"]
            self.batch_image_max_array[batch_idx] = single_data_dict["image_max"]
            self.batch_image_min_array[batch_idx] = single_data_dict["image_min"]
            self.batch_label_array[batch_idx] = single_data_dict["label"]

        real_img, target_img, label, target_label, img_min, img_max = split_multiclass_into_batch(
            self.batch_image_array, self.batch_label_array,
            self.batch_image_min_array, self.batch_image_max_array
        )
        if self.include_min_max:
            return (real_img, target_img), (label, target_label), (img_min, img_max)
        else:
            return (real_img, target_img), (label, target_label)

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(
            f"Total data num {data_num}")
