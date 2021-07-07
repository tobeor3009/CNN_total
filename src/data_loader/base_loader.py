from abc import abstractmethod
from copy import deepcopy
import cv2
import progressbar
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle as syncron_shuffle
import albumentations as A


class BaseDataGetter():

    def __init__(self):
        self.image_path_dict = None
        self.data_on_memory_dict = None
        self.on_memory = False
        self.data_len = None
        self.data_index_dict = None

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_path_dict)

        return self.data_len

    def shuffle(self):
        data_index_list = syncron_shuffle(self.data_index_dict)
        for index, shuffled_index in enumerate(data_index_list):
            self.data_index_dict[index] = shuffled_index

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
            self.data_on_memory_dict[index] = single_data_dict
            progressbar_displayed.update(value=index + 1)

        self.single_data_dict = deepcopy(self.single_data_dict)
        self.on_memory = True
        progressbar_displayed.finish()


class BaseDataLoader(tensorflow.keras.utils.Sequence):

    def __len__(self):
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()

    @abstractmethod
    def print_data_info(self):
        pass


class PreprocessPolicy():

    def __init__(self, preprocess_input):

        if preprocess_input is None:
            self.preprocess_method = lambda image_array: image_array
        elif preprocess_input == "-1~1":
            self.preprocess_method = \
                lambda image_array: (image_array / 127.5) - 1
        elif preprocess_input == "mask":
            self.preprocess_method = lambda image_array: image_array / 255
        else:
            self.preprocess_method = preprocess_input

    def __call__(self, image_array):
        image_preprocessed_array = self.preprocess_method(image_array)
        return image_preprocessed_array


class ResizePolicy():

    def __init__(self, target_size, interpolation):

        # cv2 resize policy
        interpolation_dict = {
            "bilinear": cv2.INTER_LINEAR
        }

        if target_size is None:
            self.resize_method = lambda image_array: image_array
        else:
            self.resize_method = lambda image_array: cv2.resize(src=image_array,
                                                                dsize=target_size,
                                                                interpolation=interpolation_dict[interpolation]
                                                                )

    def __call__(self, image_array):
        image_resized_array = self.resize_method(image_array)
        if len(image_resized_array.shape) == 2:
            image_resized_array = np.expand_dims(image_resized_array, axis=-1)
        return image_resized_array


class CategorizePolicy():

    def __init__(self, class_mode, num_classes, dtype):

        if class_mode == "binary":
            self.categorize_method = lambda label: label
        elif class_mode == "categorical":
            self.categorize_method = \
                lambda label: to_categorical(label, num_classes, dtype=dtype)

    def __call__(self, label):
        label = self.categorize_method(label)
        return label


class ClassifiyArgumentationPolicy():
    def __init__(self, argumentation_proba):

        positional_transform = A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1),
            A.RandomRotate90(p=1)
        ], p=0.5)

        noise_transform = A.OneOf([
            A.GaussNoise(var_limit=(0.01, 1), p=1),
        ], p=0.5)

        brightness_value = 0.05
        brightness_contrast_transform = A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-brightness_value, brightness_value), contrast_limit=(-brightness_value, brightness_value), p=1),
        ], p=0.5)

        final_transform = A.Compose([
            positional_transform,
            noise_transform,
            brightness_contrast_transform,
        ], p=argumentation_proba)

        if argumentation_proba:
            self.transform = lambda image_array: \
                final_transform(image=image_array)['image']
        else:
            self.transform = lambda image_array: image_array

    def __call__(self, image_array):
        image_transformed_array = self.transform(image_array)
        return image_transformed_array


class SegArgumentationPolicy():
    def __init__(self, argumentation_proba):

        positional_transform = A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1),
            A.RandomRotate90(p=1)
        ], p=0.5)

        noise_transform = A.OneOf([
            A.GaussNoise(var_limit=(0.01, 1), p=1),
        ], p=0.5)

        brightness_value = 0.05
        brightness_contrast_transform = A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-brightness_value, brightness_value), contrast_limit=(-brightness_value, brightness_value), p=1),
        ], p=0.5)

        self.final_transform = A.Sequential([
            positional_transform,
            noise_transform,
            brightness_contrast_transform,
        ],
            p=argumentation_proba)

        if argumentation_proba:
            self.transform = self.image_mask_sync_transform
        else:
            self.transform = lambda image_array, mask_array: \
                (image_array, mask_array)

    def __call__(self, image_array, mask_array):
        image_transformed_array, mask_transformed_array = \
            self.transform(image_array, mask_array)
        return image_transformed_array, mask_transformed_array

    def image_mask_sync_transform(self, image_array, mask_array):

        transformed = self.final_transform(image=image_array, mask=mask_array)
        image_transformed_array = transformed["image"]
        mask_transformed_array = transformed["mask"]

        return image_transformed_array, mask_transformed_array
