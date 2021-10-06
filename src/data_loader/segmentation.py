# base module
from copy import deepcopy
from tqdm import tqdm
import os
# external module
import numpy as np
import progressbar

# this library module
from .utils import imread, LazyDict, get_array_dict_lazy, get_npy_array
from .base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, SegArgumentationPolicy, \
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
"""


class SegDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 on_memory,
                 argumentation_proba,
                 argumentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 mask_preprocess_input,
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

        self.image_channel = image_channel_dict["image"]
        self.mask_channel = image_channel_dict["mask"]

        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "mask_array": None}
        self.argumentation_method = SegArgumentationPolicy(
            0, argumentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(None)
        self.mask_preprocess_method = PreprocessPolicy(None)
        if self.on_memory is True:
            self.get_data_on_ram()
        # else:
        #     self.get_data_on_disk()

        self.argumentation_method = SegArgumentationPolicy(
            argumentation_proba, argumentation_policy_dict)
        self.image_preprocess_method = PreprocessPolicy(preprocess_input)
        self.mask_preprocess_method = PreprocessPolicy(mask_preprocess_input)

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

            image_array = imread(image_path, channel=self.image_channel)
            mask_array = imread(mask_path, channel=self.mask_channel)

            image_array = self.resize_method(image_array)
            mask_array = self.resize_method(mask_array)

            image_array, mask_array = \
                self.argumentation_method(image_array, mask_array)

            image_array = self.image_preprocess_method(image_array)
            mask_array = self.mask_preprocess_method(mask_array)

            if self.is_cached is False:
                self.single_data_dict = deepcopy(self.single_data_dict)
                self.is_cached = None not in self.data_on_memory_dict.values()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["mask_array"] = mask_array
        return self.single_data_dict

    def get_data_on_disk(self):

        single_data_dict = self[0]
        image_array_shape = list(single_data_dict["image_array"].shape)
        image_array_shape = tuple([len(self)] + image_array_shape)
        image_array_dtype = single_data_dict["image_array"].dtype

        mask_array_shape = list(single_data_dict["mask_array"].shape)
        mask_array_shape = tuple([len(self)] + mask_array_shape)
        mask_array_dtype = single_data_dict["mask_array"].dtype

        # get_npy_array(path, target_size, data_key, shape, dtype)
        image_memmap_array, image_lock_path = get_npy_array(path=self.image_path_dict[0],
                                                            target_size=self.target_size,
                                                            data_key="image",
                                                            shape=image_array_shape,
                                                            dtype=image_array_dtype)
        mask_memmap_array, mask_lock_path = get_npy_array(path=self.image_path_dict[0],
                                                          target_size=self.target_size,
                                                          data_key="mask",
                                                          shape=mask_array_shape,
                                                          dtype=mask_array_dtype)

        if os.path.exists(image_lock_path) and os.path.exists(mask_lock_path):
            pass
        else:
            for index, single_data_dict in tqdm(enumerate(self)):
                image_array, mask_array = single_data_dict.values()
                image_memmap_array[index] = image_array
                mask_memmap_array[index] = mask_array

            with open(image_lock_path, "w") as _, open(mask_lock_path, "w") as _:
                pass
        array_dict_lazy = get_array_dict_lazy(key_tuple=("image_array", "mask_array"),
                                              array_tuple=(image_memmap_array, mask_memmap_array))
        self.data_on_memory_dict = LazyDict({
            i: (array_dict_lazy, i) for i in range(len(self))
        })
        self.on_memory = True


class SegDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 on_memory=False,
                 argumentation_proba=None,
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 image_channel_dict={"image": "rgb", "mask": None},
                 preprocess_input="-1~1",
                 mask_preprocess_input="mask",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         on_memory=on_memory,
                                         argumentation_proba=argumentation_proba,
                                         argumentation_policy_dict=argumentation_policy_dict,
                                         image_channel_dict=image_channel_dict,
                                         preprocess_input=preprocess_input,
                                         mask_preprocess_input=mask_preprocess_input,
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
                 argumentation_policy_dict,
                 image_channel_dict,
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

        self.image_channel = image_channel_dict["image"]
        self.mask_channel = image_channel_dict["mask"]

        self.target_size = target_size
        self.interpolation = interpolation

        self.resize_method = ResizePolicy(target_size, interpolation)

        self.is_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image_array": None, "mask_array": None}
        if self.on_memory is True:
            self.argumentation_method = \
                SegArgumentationPolicy(0, argumentation_policy_dict)
            self.image_preprocess_method = PreprocessPolicy(None)
            self.mask_preprocess_method = PreprocessPolicy(None)
            self.get_data_on_ram()

        self.argumentation_method = \
            SegArgumentationPolicy(argumentation_proba,
                                   argumentation_policy_dict)
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

    def get_data_on_ram(self):
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
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 image_channel_dict={"image": "rgb", "mask": None},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SelfModifyDataGetter(image_path_list=image_path_list,
                                                mask_path_list=mask_path_list,
                                                on_memory=on_memory,
                                                argumentation_proba=argumentation_proba,
                                                argumentation_policy_dict=argumentation_policy_dict,
                                                image_channel_dict=image_channel_dict,
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
