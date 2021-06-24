# base module

# external module
import cv2
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy

"""
Expect Data Path Structure

train - class_names
valid - class_names
test - class_names

Example)
train - non_tumor
      - tumor
label_dict = {"non_tumor":0, "tumor":1}
"""

"""
Test needed(class cache Working?)
"""

class ClassifyDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_path_list = image_path_list
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.categorize_method = CategorizePolicy(class_mode, self.num_classes)
        self.preprocess_method = PreprocessPolicy(preprocess_input)
        self.resize_method = ResizePolicy(target_size, interpolation)

        self.cached_class_no = 0
        self.is_class_cached = False
        self.data_index = -1 * np.ones((len(self), ), dtype="int32")

        self.classes = np.zeros((len(self), ), dtype=dtype)
        self.classes = self.categorize_method(self.classes)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]

        image_array = imread(image_path, channel="rgb")
        image_array = self.resize_method(image_array)
        image_array = self.preprocess_method(image_array)

        if self.is_class_cached:
            label = self.classes[i]
        else:
            image_dir_name = get_parent_dir_name(image_path)
            label = self.label_to_index_dict[image_dir_name]
            label = self.categorize_method(label)
            self.classes[i] = label
            self.cached_class_no += 1
            self.is_class_cached = self.cached_class_no == len(self)

        return image_array, label

    def shuffle(self):
        self.image_path_list, self.classes = \
            syncron_shuffle(self.image_path_list, self.classes)


class ClassifyDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
                                              label_to_index_dict=label_to_index_dict,
                                              preprocess_input=preprocess_input,
                                              target_size=target_size,
                                              interpolation=interpolation,
                                              class_mode=class_mode,
                                              dtype=dtype
                                              )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape
        if target_size is None:
            target_size = self.source_data_shape[:2]
            self.data_getter.target_size = target_size
            self.source_data_shape[:2] = target_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.zeros(
            (self.batch_size, *self.source_data_shape), dtype=self.dtype)
        if self.class_mode == "binary":
            batch_y = np.zeros(
                (self.batch_size, ), dtype=self.dtype)
        elif self.class_mode == "categorical":
            batch_y = np.zeros(
                (self.batch_size, self.num_classes), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]
            batch_y[batch_index] = data[1]

        return batch_x, batch_y

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
