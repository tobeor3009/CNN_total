# base module

# external module
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy

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
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode):
        super().__init__()

        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy("mask")
        self.resize_method = ResizePolicy(target_size, interpolation)
        self.categorize_method = CategorizePolicy(class_mode, self.num_classes)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        mask_path = self.mask_path_list[i]

        image_array = imread(image_path, channel="rgb")
        mask_array = imread(mask_path)

        image_array = self.resize_method(image_array)
        mask_array = self.resize_method(mask_array)

        image_array = self.image_preprocess_method(image_array)
        mask_array = self.mask_preprocess_method(mask_array)

        image_dir_name = get_parent_dir_name(image_path)
        label = self.label_to_index_dict[image_dir_name]
        label = self.categorize_method(label)

        preserve = np.mean(mask_array)

        single_data_dict = {"image_array": image_array,
                            "mask_array": mask_array,
                            "label": label,
                            "preserve": preserve}

        return single_data_dict

    def shuffle(self):
        self.image_path_list, self.mask_path_list = \
            syncron_shuffle(self.image_path_list, self.mask_path_list)


class MultiLabelDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32",
                 class_mode="binary"):
        self.data_getter = MultiLabelDataGetter(image_path_list=image_path_list,
                                                mask_path_list=mask_path_list,
                                                label_to_index_dict=label_to_index_dict,
                                                preprocess_input=preprocess_input,
                                                target_size=target_size,
                                                interpolation=interpolation,
                                                class_mode=class_mode
                                                )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape

        self.shuffle = shuffle
        self.dtype = dtype
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty(
            (self.batch_size, *self.source_data_shape), dtype=self.dtype)
        batch_y = np.empty(
            (self.batch_size, self.num_classes), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            batch_x[batch_index] = single_data_dict[0]
            batch_y[batch_index] = single_data_dict[1]

        return batch_x, batch_y

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
