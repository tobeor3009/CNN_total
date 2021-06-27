# base module

# external module
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ArgumentationPolicy

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

"""
To Be Done:
    - Test class cache
"""


class ClassifyDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 on_memory,
                 preprocess_input,
                 target_size,
                 interpolation,
                 argumentation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.data_on_memory_dict = {}
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.on_memory = on_memory
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)
        self.argumentation_method = ArgumentationPolicy(
            argumentation, "classfication")
        self.preprocess_method = PreprocessPolicy(preprocess_input)
        self.categorize_method = CategorizePolicy(
            class_mode, self.num_classes, dtype)

        self.cached_class_no = 0
        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.class_dict = {i: self.categorize_method(
            0) for i in range(len(self))}

        if self.on_memory:
            self.get_data_on_memory()

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            single_data_tuple = self.data_on_memory_dict[current_index]
        else:
            image_path = self.image_path_dict[current_index]
            image_array = imread(image_path, channel="rgb")
            image_array = self.resize_method(image_array)
            image_array = self.argumentation_method(image_array)
            image_array = self.preprocess_method(image_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
            else:
                image_dir_name = get_parent_dir_name(image_path)
                label = self.label_to_index_dict[image_dir_name]
                label = self.categorize_method(label)
                self.class_dict[current_index] = label
                self.cached_class_no += 1
                self.is_class_cached = self.cached_class_no == len(self)

            single_data_tuple = image_array, label

        return single_data_tuple


class ClassifyDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 on_memory=False,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 argumentation=None,
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
                                              label_to_index_dict=label_to_index_dict,
                                              on_memory=on_memory,
                                              preprocess_input=preprocess_input,
                                              target_size=target_size,
                                              interpolation=interpolation,
                                              argumentation=argumentation,
                                              class_mode=class_mode,
                                              dtype=dtype
                                              )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.data_getter.cached_class_no = 0
        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = min(start + self.batch_size, len(self.data_getter))
        current_batch_size = end - start
        batch_x = np.zeros(
            (current_batch_size, *self.source_data_shape), dtype=self.dtype)
        batch_y = np.zeros((current_batch_size, ), dtype=self.dtype)
        batch_y = self.data_getter.categorize_method(batch_y)

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_tuple = self.data_getter[total_index]
            batch_x[batch_index] = single_data_tuple[0]
            batch_y[batch_index] = single_data_tuple[1]

        return batch_x, batch_y

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
