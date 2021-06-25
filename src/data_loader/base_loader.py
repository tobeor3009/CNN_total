from abc import abstractmethod
from albumentations.augmentations.transforms import VerticalFlip
import cv2
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

        self.on_memory = False

        for index, single_data_tuple in enumerate(self):
            if index >= len(self):
                break
            self.data_on_memory_dict[index] = single_data_tuple
            
        self.on_memory = True


class BaseDataLoader(tensorflow.keras.utils.Sequence):

    def __len__(self):
        """Denotes the number of batches per epoch"""
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
            self.preprocess_method = lambda x: (x / 127.5) - 1
        elif preprocess_input == "mask":
            self.preprocess_method = lambda x: np.expand_dims(x / 255, axis=-1)
        else:
            self.preprocess_method = preprocess_input

    def __call__(self, image_array):
        image_resized_array = self.preprocess_method(image_array)
        return image_resized_array


class ResizePolicy():

    def __init__(self, target_size, interpolation):

        # cv2 resize policy
        interpolation_dict = {
            "bilinear": cv2.INTER_LINEAR
        }

        if target_size is None:
            self.resize_method = lambda x: x
        else:
            self.resize_method = lambda x: cv2.resize(src=x,
                                                      dsize=target_size,
                                                      interpolation=interpolation_dict[interpolation]
                                                      )

    def __call__(self, image_array):
        image_resized_array = self.resize_method(image_array)
        return image_resized_array


class CategorizePolicy():

    def __init__(self, class_mode, num_classes, dtype):

        if class_mode == "binary":
            self.categorize_method = lambda x: x
        elif class_mode == "categorical":
            self.categorize_method = lambda x: to_categorical(
                x, num_classes, dtype=dtype)

    def __call__(self, label):
        label = self.categorize_method(label)
        return label


class ArgumentationPolicy():
    def __init__(self, argumentation, task):

        positional_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

        brightness_contrast_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
        ])

        noise_transform = A.Compose([
            A.Blur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
        ])

        classification_transform = A.Compose([
            positional_transform,
            brightness_contrast_transform,
            noise_transform,
        ])

        if task == "classfication":
            final_transform = classification_transform

        if argumentation is None:
            self.transform = lambda x: x
        else:
            self.transform = A.Compose([
                final_transform
            ], p=0.9)

    def __call__(self, image_array):
        image_transformed_array = self.transform(image_array)
        return image_transformed_array
