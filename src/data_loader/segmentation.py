# base module

# external module
import numpy as np

# this library module
from .utils import imread
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, ArgumentationPolicy

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
"""


class SegDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 on_memory,
                 preprocess_input,
                 target_size,
                 interpolation,
                 argumentation
                 ):
        super().__init__()

        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.data_on_memory_dict = {}
        self.on_memory = on_memory
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)
        self.argumentation_method = ArgumentationPolicy(
            argumentation, "segmentation")
        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy("mask")

        self.data_index_dict = {i: i for i in range(len(self))}

        if self.on_memory:
            self.get_data_on_memory()
        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"

    def __getitem__(self, i):

        current_index = self.data_index_dict[i]

        if self.on_memory:
            single_data_tuple = self.data_on_memory_dict[current_index]
        else:
            image_path = self.image_path_list[current_index]
            mask_path = self.mask_path_list[current_index]

            image_array = imread(image_path, channel="rgb")
            mask_array = imread(mask_path)

            image_array = self.resize_method(image_array)
            mask_array = self.resize_method(mask_array)

            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)

            single_data_tuple = image_array, mask_array

        return single_data_tuple


class SegDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 on_memory=False,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 argumentation=None,
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         on_memory=on_memory,
                                         preprocess_input=preprocess_input,
                                         target_size=target_size,
                                         interpolation=interpolation,
                                         argumentation=argumentation
                                         )
        self.batch_size = batch_size
        self.dtype = dtype
        self.image_data_shape = self.data_getter[0][0].shape
        self.mask_data_shape = self.data_getter[0][1].shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        batch_y = np.empty(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            single_data_tuple = self.data_getter[total_index]
            batch_x[batch_index] = single_data_tuple[0]
            batch_y[batch_index] = single_data_tuple[1]

        return batch_x, batch_y

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
