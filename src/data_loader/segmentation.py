# base module

# external module
import cv2
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread
from .base_loader import BaseDataGetter, BaseDataLoader

"""
Expect Data Path Structure

train - image
      - mask
valid - image
      - mask
test  - image
      - mask
or

"""
BACKBONE = "inceptionv3"


class SegDataGetter(BaseDataGetter):
    def __init__(self,
                 image_path_list,
                 mask_path_list,
                 preprocess_input,
                 target_size,
                 interpolation
                 ):
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        mask_path = self.mask_path_list[i]

        image_array = imread(image_path, channel="rgb")
        mask_array = imread(mask_path)

        # normalize image: [0, 255] => [-1, 1]
        # normalize mask: [0, 255] => [0, 1]
        if self.interpolation == "bilinear":
            image_array = cv2.resize(
                image_array, self.target_size, cv2.INTER_LINEAR)
            mask_array = cv2.resize(
                mask_array, self.target_size, cv2.INTER_LINEAR)
            mask_array = np.expand_dims(mask_array, axis=-1)

        if self.preprocess_input:
            image_array = self.preprocess_input(image_array)
        else:
            image_array = (image_array / 127.5) - 1

        mask_array = (mask_array / 255)

        return image_array, mask_array

    def shuffle(self):
        self.image_path_list, self.mask_path_list = \
            syncron_shuffle(self.image_path_list, self.mask_path_list)


class SegDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = SegDataGetter(image_path_list=image_path_list,
                                         mask_path_list=mask_path_list,
                                         preprocess_input=preprocess_input,
                                         target_size=target_size,
                                         interpolation=interpolation
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
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]
            batch_y[batch_index] = data[1]

        return batch_x, batch_y
