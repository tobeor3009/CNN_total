import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
import cv2


class BaseDataGetter():

    def __init__(self):
        self.data_len = None

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_path_list)

        return self.data_len


class BaseDataLoader(tensorflow.keras.utils.Sequence):

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()


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

    def __init__(self, class_mode, num_classes):

        if class_mode == "binary":
            self.categorize_method = lambda x: x
        elif class_mode == "categorical":
            self.categorize_method = lambda x: to_categorical(x, num_classes)

    def __call__(self, label):
        label = self.categorize_method(label)
        return label
