from abc import abstractmethod
from copy import deepcopy
import cv2
import progressbar
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle as syncron_shuffle
import albumentations as A

base_augmentation_policy_dict = {
    "positional": True,
    "noise": True,
    "elastic": True,
    "randomcrop": False,
    "brightness_contrast": True,
    "hist": False,
    "color": True,
    "to_jpeg": True
}

positional_transform = A.OneOf([
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.RandomRotate90(p=1)
], p=1)

noise_transform = A.OneOf([
    A.GaussNoise(var_limit=(0.01, 5), p=1),
    A.Downscale(always_apply=False, p=0.5,
                scale_min=0.699999988079071, scale_max=0.9900000095367432,
                interpolation=2),

], p=1)

elastic_tranform = A.OneOf([A.ElasticTransform(always_apply=False, p=0.5,
                                               alpha=0.20000000298023224, sigma=3.359999895095825,
                                               alpha_affine=2.009999990463257, interpolation=1, border_mode=1,
                                               value=(0, 0, 0), mask_value=None, approximate=False),
                            A.GridDistortion(always_apply=False, p=0.5, num_steps=1,
                                             distort_limit=(-0.029999999329447746,
                                                            0.05000000074505806),
                                             interpolation=2, border_mode=0, value=(0, 0, 0), mask_value=None)
                            ])

brightness_value = 0.02
brightness_contrast_transform = A.OneOf([
    A.RandomBrightnessContrast(brightness_limit=(-brightness_value, brightness_value),
                               contrast_limit=(-brightness_value,
                                               brightness_value),
                               p=1),
], p=1)

hist_transform = A.OneOf([
    A.CLAHE(always_apply=False, p=0.5,
            clip_limit=(1, 15), tile_grid_size=(8, 8)),
    # A.Equalize(always_apply=False, p=0.5,
    #            mode='cv', by_channels=False),
], p=1)

color_transform = A.OneOf([
    A.ISONoise(always_apply=False, p=0.5,
               intensity=(0.05000000074505806, 0.12999999523162842),
               color_shift=(0.009999999776482582, 0.26999998092651367)),
    A.HueSaturationValue(p=0.1)
], p=1)

to_jpeg_transform = A.ImageCompression(quality_lower=10,
                                       quality_upper=25, p=1)


def identity_fn(any):
    return any


def identity_multi_fn(image_array, mask_array):
    return image_array, mask_array


def normalize_image(image_array):
    return (image_array / 127.5) - 1


def normalize_mask(mask_array):
    mask_array = (mask_array // 255).astype("float32")
    return mask_array


def to_tuple(int_or_tuple):
    if isinstance(int_or_tuple, int):
        int_or_tuple = (int_or_tuple, int_or_tuple)
    return int_or_tuple


class BaseDataGetter():

    def __init__(self):
        self.image_path_dict = None
        self.data_on_ram_dict = None
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
            self.data_on_ram_dict[index] = single_data_dict
            progressbar_displayed.update(value=index + 1)

        self.single_data_dict = deepcopy(self.single_data_dict)
        self.on_memory = True
        progressbar_displayed.finish()

    def check_class_dict_cached(self):
        for _, value in self.class_dict.items():
            if value is None:
                return False
        return True


class BaseDataLoader(tensorflow.keras.utils.Sequence):

    def __len__(self):
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()

    @abstractmethod
    def print_data_info(self):
        pass


class BaseIterDataLoader():

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
            self.preprocess_method = identity_fn
        elif preprocess_input == "-1~1":
            self.preprocess_method = normalize_image
        elif preprocess_input == "mask" or preprocess_input == "0~1":
            self.preprocess_method = normalize_mask
        else:
            self.preprocess_method = preprocess_input

    def __call__(self, image_array):
        image_preprocessed_array = self.preprocess_method(image_array)
        return image_preprocessed_array


class ResizePolicy():

    def __init__(self, target_size, interpolation):

        # cv2 resize policy
        interpolation_dict = {
            "bilinear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC
        }
        self.target_size = to_tuple(target_size)
        self.interpolation = interpolation_dict[interpolation]
        if target_size is None:
            self.resize_method = identity_fn
        else:
            self.resize_method = self.resize_fn

    def __call__(self, image_array):
        image_resized_array = self.resize_method(image_array)
        if len(image_resized_array.shape) == 2:
            image_resized_array = np.expand_dims(image_resized_array, axis=-1)
        return image_resized_array

    def resize_fn(self, image_array):
        resized_array = cv2.resize(src=image_array,
                                   dsize=self.target_size,
                                   interpolation=self.interpolation
                                   )
        return resized_array


class CategorizePolicy():

    def __init__(self, class_mode, num_classes, dtype):
        self.num_classes = num_classes
        self.dtype = dtype
        if class_mode == "binary":
            self.categorize_method = identity_fn
        elif class_mode == "categorical":
            self.categorize_method = self.one_hot_fn

    def __call__(self, label):
        label = self.categorize_method(label)
        return label

    def one_hot_fn(self, label):
        label_array = to_categorical(label,
                                     self.num_classes, dtype=self.dtype)
        return label_array


class ClassifyaugmentationPolicy():
    def __init__(self,
                 augmentation_proba,
                 augmentation_policy_dict):

        final_transform_list = []
        if augmentation_policy_dict["randomcrop"]:
            randomcrop_transform = A.RandomCrop(
                *augmentation_policy_dict["randomcrop"], p=1)
            final_transform_list.append(randomcrop_transform)
        if augmentation_policy_dict["noise"] is True:
            final_transform_list.append(noise_transform)
        if augmentation_policy_dict["elastic"] is True:
            final_transform_list.append(elastic_tranform)
        if augmentation_policy_dict["brightness_contrast"] is True:
            final_transform_list.append(brightness_contrast_transform)
        if augmentation_policy_dict["hist"] is True:
            final_transform_list.append(hist_transform)
        if augmentation_policy_dict["color"] is True:
            final_transform_list.append(color_transform)
        if augmentation_policy_dict["to_jpeg"] is True:
            final_transform_list.append(to_jpeg_transform)

        transform_1 = positional_transform
        transform_2 = A.OneOf(final_transform_list, p=1)

        self.final_transform = A.Compose([transform_1, transform_2],
                                         p=augmentation_proba)
        if augmentation_proba:
            self.transform = self.image_transform
        else:
            self.transform = identity_fn

    def __call__(self, image_array):
        image_transformed_array = self.transform(image_array)
        return image_transformed_array

    def image_transform(self, image_array):
        if image_array.ndim == 3:
            image_transformed_array = self.image_transform_2d(image_array)
        else:
            image_transformed_array = self.image_transform_3d(image_array)
        return image_transformed_array

    def image_transform_2d(self, image_array):

        transformed = self.final_transform(image=image_array)
        image_transformed_array = transformed["image"]

        return image_transformed_array

    def image_transform_3d(self, image_array):

        image_transformed_array = np.empty(
            image_array.shape, dtype=image_array.dtype)
        for slice_idx, image_slice_array in enumerate(image_array):
            image_transformed_slice_array = self.image_transform_2d(
                image_slice_array)
            image_transformed_array[slice_idx] = image_transformed_slice_array
        return image_transformed_array


class SegaugmentationPolicy():
    def __init__(self,
                 augmentation_proba,
                 augmentation_policy_dict):

        final_transform_list = []
        if augmentation_policy_dict["randomcrop"]:
            randomcrop_transform = A.RandomCrop(*augmentation_policy_dict["randomcrop"],
                                                p=1)
            final_transform_list.append(randomcrop_transform)
        if augmentation_policy_dict["noise"] is True:
            final_transform_list.append(noise_transform)
        if augmentation_policy_dict["elastic"] is True:
            final_transform_list.append(elastic_tranform)
        if augmentation_policy_dict["brightness_contrast"] is True:
            final_transform_list.append(brightness_contrast_transform)
        if augmentation_policy_dict["hist"] is True:
            final_transform_list.append(hist_transform)
        if augmentation_policy_dict["color"] is True:
            final_transform_list.append(color_transform)
        if augmentation_policy_dict["to_jpeg"] is True:
            final_transform_list.append(to_jpeg_transform)

        transform_1 = positional_transform
        transform_2 = A.OneOf(final_transform_list, p=1)

        self.final_transform = A.Compose([transform_1, transform_2],
                                         p=augmentation_proba)

        if augmentation_proba:
            self.transform = self.image_mask_sync_transform
        else:
            self.transform = identity_multi_fn

    def __call__(self, image_array, mask_array):
        image_transformed_array, mask_transformed_array = \
            self.transform(image_array, mask_array)
        return image_transformed_array, mask_transformed_array

    def image_mask_sync_transform(self, image_array, mask_array):

        transformed = self.final_transform(image=image_array, mask=mask_array)
        image_transformed_array = transformed["image"]
        mask_transformed_array = transformed["mask"]

        return image_transformed_array, mask_transformed_array
