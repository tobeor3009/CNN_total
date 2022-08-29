# base module
from copy import deepcopy
from glob import glob
import math
# external module
import numpy as np
import albumentations as A

# this library module
from .utils import imread, get_parent_dir_name, SingleProcessPool, MultiProcessPool, lazy_cycle
from .base_loader import BaseDataGetter, BaseIterDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyaugmentationPolicy, \
    base_augmentation_policy_dict

positional_transform = A.OneOf([
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.RandomRotate90(p=1)
], p=0.5)

noise_transform = A.OneOf([
    A.Blur(blur_limit=(2, 2), p=1),
    A.GaussNoise(var_limit=(0.01, 5), p=1),
], p=0.5)

brightness_value = 0.1
brightness_contrast_transform = A.OneOf([
    A.RandomBrightnessContrast(brightness_limit=(-brightness_value, brightness_value),
                               contrast_limit=(-brightness_value,
                                               brightness_value),
                               p=1),
], p=0.5)

position_proba = 1 - (0.5) ** (0.25)


def get_transform_image(image, keypoints, target_shape, augmentation_proba):
    transform = A.Compose([A.Resize(target_shape[0], target_shape[1], 1, always_apply=True),
                           positional_transform,
                          noise_transform,
                          brightness_contrast_transform],
                          p=augmentation_proba,
                          keypoint_params=A.KeypointParams(format='xy',
                                                           remove_invisible=False,
                                                           angle_in_degrees=False))
    transformed = transform(image=image, keypoints=keypoints)
    transformed_image = transformed['image']
    transformed_keypoints = np.array(transformed['keypoints'])
    transformed_keypoints[:, 0] = transformed_keypoints[:, 0] / target_shape[1]
    transformed_keypoints[:, 1] = transformed_keypoints[:, 1] / target_shape[0]
    return transformed_image, transformed_keypoints


def pose_estimation_collate_fn(data_object_list):
    batch_image_array = []
    batch_keypoint_array = []
    batch_label_array = []
    for single_data_dict in data_object_list:
        image_array = single_data_dict["image"]
        keypoint_array = single_data_dict["keypoints"]
        label_array = single_data_dict["label"]
        batch_image_array.append(image_array)
        batch_keypoint_array.append(keypoint_array)
        batch_label_array.append(label_array)
    batch_image_array = np.stack(batch_image_array, axis=0)
    batch_keypoint_array = np.stack(batch_keypoint_array, axis=0)
    batch_label_array = np.stack(batch_label_array, axis=0)

    return batch_image_array, (batch_keypoint_array, batch_label_array)


def keypoint_collate_fn(image_id, total_annotation_dict):
    image_id = str(int(image_id))
    if image_id in total_annotation_dict.keys():
        annotation_dict = total_annotation_dict[image_id][0]
        keypoint_list = annotation_dict["keypoints"]
        keypoints = []
        labels = []
        for keypoint_idx in range(17):
            x_idx, y_idx, label = keypoint_list[3 *
                                                keypoint_idx: 3 * keypoint_idx + 3]
            keypoints.append((x_idx, y_idx))
            labels.append(label)
    else:
        keypoints = [(0, 0) for _ in range(17)]
        labels = [0 for _ in range(17)]
    return np.array(keypoints), np.array(labels)


class PoseDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 total_annotation_dict,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 dtype):
        super().__init__()

        self.image_path_dict = {idx: image_folder
                                for idx, image_folder in enumerate(image_path_list)}
        self.data_on_ram_dict = {}
        self.total_annotation_dict = total_annotation_dict
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation

        self.target_shape = target_size

        self.data_index_dict = {i: i for i in range(len(self))}
        self.single_data_dict = {"image": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}

        self.augmentation_proba = augmentation_proba
        self.augmentation_method = get_transform_image

        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = self.data_index_dict[i]

        image_path = self.image_path_dict[current_index]
        image_id = get_parent_dir_name(image_path, level=0)
        image_id = image_id.split(".")[0]
        image_array = imread(image_path, channel=self.image_channel)
        keypoint_array, label_array = keypoint_collate_fn(image_id,
                                                          self.total_annotation_dict)
        image_array, keypoint_array = self.augmentation_method(image_array,
                                                               keypoint_array,
                                                               self.target_shape,
                                                               self.augmentation_proba)
        image_array = self.preprocess_method(image_array)
        keypoint_array = np.reshape(keypoint_array, (-1))
        self.single_data_dict = deepcopy(self.single_data_dict)
        self.single_data_dict["image"] = image_array
        self.single_data_dict["keypoints"] = keypoint_array
        self.single_data_dict["label"] = label_array

        return self.single_data_dict


class PoseDataloader(BaseIterDataLoader):

    def __init__(self,
                 image_path_list=None,
                 total_annotation_dict=None,
                 batch_size=None,
                 num_workers=1,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"
                 ):
        self.data_getter = PoseDataGetter(image_path_list=image_path_list,
                                          total_annotation_dict=total_annotation_dict,
                                          on_memory=on_memory,
                                          augmentation_proba=augmentation_proba,
                                          augmentation_policy_dict=augmentation_policy_dict,
                                          image_channel_dict=image_channel_dict,
                                          preprocess_input=preprocess_input,
                                          target_size=target_size,
                                          interpolation=interpolation,
                                          dtype=dtype
                                          )
        self.batch_size = batch_size
        self.data_num = len(self.data_getter)
        self.shuffle = shuffle
        self.dtype = dtype

        if num_workers == 1:
            self.data_pool = SingleProcessPool(data_getter=self.data_getter,
                                               batch_size=self.batch_size,
                                               collate_fn=pose_estimation_collate_fn,
                                               shuffle=self.shuffle
                                               )
        else:
            self.data_pool = MultiProcessPool(data_getter=self.data_getter,
                                              batch_size=self.batch_size,
                                              num_workers=num_workers,
                                              collate_fn=pose_estimation_collate_fn,
                                              shuffle=self.shuffle
                                              )

        self.print_data_info()
        self.on_epoch_end()

    def __iter__(self):
        return lazy_cycle(self.data_pool)

    def __next__(self):
        return next(self.data_pool)

    def __getitem__(self, i):

        start = i * self.batch_size
        end = min(start + self.batch_size, self.data_num)
        batch_image_array = []
        batch_keypoint_array = []
        batch_label_array = []
        for total_index in range(start, end):
            single_data_dict = self.data_getter[total_index]
            batch_image_array.append(single_data_dict["image"])
            batch_keypoint_array.append(single_data_dict["keypoints"])
            batch_label_array.append(single_data_dict["label"])
        batch_image_array = np.stack(batch_image_array, axis=0)
        batch_keypoint_array = np.stack(batch_keypoint_array, axis=0)
        batch_label_array = np.stack(batch_label_array, axis=0)
        return batch_image_array, (batch_keypoint_array, batch_label_array)

    def __len__(self):
        return math.ceil(self.data_num / self.batch_size)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num}")
