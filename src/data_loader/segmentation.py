# base module
from copy import deepcopy
# external module
import numpy as np
import progressbar

# this library module
from .utils import imread
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, SegArgumentationPolicy

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
                 argumentation_proba,
                 preprocess_input,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.mask_path_dict = {index: mask_path for index,
                               mask_path in enumerate(mask_path_list)}
        self.data_on_memory_dict = {index: None for index, _
                                    in enumerate(image_path_list)}
        self.on_memory = on_memory
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "mask_array": None}
        if self.on_memory is True:
            self.argumentation_method = SegArgumentationPolicy(0)
            self.image_preprocess_method = PreprocessPolicy(None)
            self.mask_preprocess_method = PreprocessPolicy(None)
            self.get_data_on_memory()

        self.argumentation_method = SegArgumentationPolicy(argumentation_proba)
        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy("mask")

        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array, mask_array = \
                self.data_on_memory_dict[current_index].values()
            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)
            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)
        else:
            image_path = self.image_path_dict[current_index]
            mask_path = self.mask_path_dict[current_index]

            image_array = imread(image_path, channel="rgb")
            mask_array = imread(mask_path)

            image_array = self.resize_method(image_array)
            mask_array = self.resize_method(mask_array)

            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)

            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)

            if self.is_cached is False:
                self.is_cached = None not in self.data_on_memory_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array
        return self.single_data_dict


class SegDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 on_memory=False,
                 argumentation_proba=None,
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         on_memory=on_memory,
                                         argumentation_proba=argumentation_proba,
                                         preprocess_input=preprocess_input,
                                         target_size=target_size,
                                         interpolation=interpolation
                                         )
        self.batch_size = batch_size
        self.image_data_shape = self.data_getter[0]["image_array"].shape
        self.mask_data_shape = self.data_getter[0]["mask_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_mask_array = np.zeros(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_mask_array[batch_index] = single_data_dict["mask_array"]

        return self.batch_image_array, self.batch_mask_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")


class SelfModifyDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 on_memory,
                 argumentation_proba,
                 preprocess_input,
                 target_size,
                 interpolation
                 ):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.mask_path_dict = {index: mask_path for index,
                               mask_path in enumerate(mask_path_list)}
        self.data_on_memory_dict = {index: None for index, _
                                    in enumerate(image_path_list)}
        self.on_memory = on_memory
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "mask_array": None}
        if self.on_memory is True:
            self.argumentation_method = SegArgumentationPolicy(0)
            self.image_preprocess_method = PreprocessPolicy(None)
            self.mask_preprocess_method = PreprocessPolicy(None)
            self.get_data_on_memory()

        self.argumentation_method = SegArgumentationPolicy(argumentation_proba)
        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy("mask")

        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        if self.on_memory:
            image_array = self.data_on_memory_dict[current_index]
        else:
            image_path = self.image_path_dict[current_index]
            image_array = imread(image_path, channel="rgb")
            image_array = self.resize_method(image_array)

            if self.is_cached is False:
                self.is_cached = None not in self.data_on_memory_dict.values()

        mask_path = self.mask_path_dict[current_index]
        mask_array = imread(mask_path)
        mask_array = self.resize_method(mask_array)

        image_array, mask_array = \
            self.argumentation_method(image_array, mask_array)

        image_array = self.image_preprocess_method(image_array)
        mask_array = self.mask_preprocess_method(mask_array)

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array

        return self.single_data_dict

    def get_data_on_memory(self):
        widgets = [
            ' [',
            progressbar.Counter(format=f'%(value)02d/%(max_value)d'),
            '] ',
            progressbar.Bar(),
            ' (',
            progressbar.ETA(),
            ') ',
        ]
        progressbar_displayed = progressbar.ProgressBar(widgets=widgets,
                                                        maxval=len(self)).start()

        self.on_memory = False
        for index, single_data_dict in enumerate(self):
            self.data_on_memory_dict[index] = \
                deepcopy(single_data_dict["image_array"])
            progressbar_displayed.update(value=index + 1)

        self.on_memory = True
        progressbar_displayed.finish()


class SelfModifyDataLoader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 on_memory=False,
                 argumentation_proba=None,
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SelfModifyDataGetter(image_path_list=image_path_list,
                                                mask_path_list=mask_path_list,
                                                on_memory=on_memory,
                                                argumentation_proba=argumentation_proba,
                                                preprocess_input=preprocess_input,
                                                target_size=target_size,
                                                interpolation=interpolation
                                                )
        self.batch_size = batch_size
        self.image_data_shape = self.data_getter[0]["image_array"].shape
        self.mask_data_shape = self.data_getter[0]["mask_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_mask_array = np.zeros(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict = self.data_getter[total_index]
            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_mask_array[batch_index] = single_data_dict["mask_array"]

        return self.batch_image_array, self.batch_mask_array

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
