# base module
from copy import deepcopy
from tqdm import tqdm
# external module
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from ..utils import imread, get_parent_dir_name
from ..base.base_loader import BaseDataGetter, BaseDataLoader, \
    ResizePolicy, PreprocessPolicy, CategorizePolicy, ClassifyAugmentationPolicy, \
    base_augmentation_policy_dict

class StarGanDataGetter(BaseDataGetter):

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 label_level,
                 on_memory,
                 augmentation_proba,
                 augmentation_policy_dict,
                 image_channel_dict,
                 preprocess_input,
                 target_size,
                 interpolation,
                 class_mode,
                 dtype):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.data_on_ram_dict = {}
        self.label_to_index_dict = label_to_index_dict
        self.label_level = label_level
        self.num_classes = len(self.label_to_index_dict)
        self.on_memory = on_memory
        self.image_channel = image_channel_dict["image"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.resize_method = ResizePolicy(target_size, interpolation)
        self.categorize_method = \
            CategorizePolicy(class_mode, self.num_classes, dtype)

        self.is_class_cached = False
        self.data_index_dict = {i: i for i in range(len(self))}
        self.target_data_index_list = list(range(len(self)))
        self.target_data_index_list = \
            syncron_shuffle(self.target_data_index_list)
        self.target_data_index_quque_len = len(self)

        self.single_data_dict = {"image_array": None, "label": None}
        self.target_single_data_dict = {"image_array": None, "label": None}
        self.class_dict = {i: None for i in range(len(self))}

        self.augmentation_method = ClassifyAugmentationPolicy(
            0, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(None)

        if self.on_memory is True:
            self.get_data_on_ram()

        self.augmentation_method = \
            ClassifyAugmentationPolicy(
                augmentation_proba, augmentation_policy_dict)
        self.preprocess_method = PreprocessPolicy(preprocess_input)

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        if self.target_data_index_quque_len != 0:
            self.target_data_index_quque_len -= 1
        else:
            self.target_data_index_quque_len = len(self) - 1
            self.target_data_index_list = syncron_shuffle(
                self.target_data_index_list)

        current_index = self.data_index_dict[i]
        target_index = self.target_data_index_list[i]

        if self.on_memory:
            image_array, label = \
                self.data_on_ram_dict[current_index].values()
            target_image_array, target_label = \
                self.data_on_ram_dict[target_index].values()
            image_array = self.augmentation_method(image_array)
            target_image_array = self.augmentation_method(target_image_array)

            image_array = self.preprocess_method(image_array)
            target_image_array = self.preprocess_method(target_image_array)
        else:
            image_path = self.image_path_dict[current_index]
            target_image_path = self.image_path_dict[target_index]

            image_array = imread(image_path, channel=self.image_channel)
            target_image_array = imread(
                target_image_path, channel=self.image_channel)

            image_array = self.resize_method(image_array)
            target_image_array = self.resize_method(target_image_array)

            image_array = self.augmentation_method(image_array)
            target_image_array = self.augmentation_method(target_image_array)

            image_array = self.preprocess_method(image_array)
            target_image_array = self.preprocess_method(target_image_array)

            if self.is_class_cached:
                label = self.class_dict[current_index]
                target_label = self.class_dict[target_index]
            else:
                image_dir_name = get_parent_dir_name(
                    image_path, self.label_level)
                target_image_dir_name = get_parent_dir_name(
                    target_image_path, self.label_level)

                label = self.label_to_index_dict[image_dir_name]
                target_label = self.label_to_index_dict[target_image_dir_name]

                label = self.categorize_method(label)
                target_label = self.categorize_method(target_label)

                self.class_dict[current_index] = label
                self.class_dict[target_index] = target_label

                self.single_data_dict = deepcopy(self.single_data_dict)
                self.target_single_data_dict = deepcopy(
                    self.target_single_data_dict)
                self.is_class_cached = self.check_class_dict_cached()

        self.single_data_dict["image_array"] = image_array
        self.single_data_dict["label"] = label
        self.target_single_data_dict["image_array"] = target_image_array
        self.target_single_data_dict["label"] = target_label

        return self.single_data_dict, self.target_single_data_dict

    def get_data_on_ram(self):

        self.on_memory = False

        image_range = range(self.data_len)

        for index in tqdm(image_range):
            image_path = self.image_path_dict[index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = self.resize_method(image_array)

            image_dir_name = get_parent_dir_name(image_path, self.label_level)
            label = self.label_to_index_dict[image_dir_name]
            label = self.categorize_method(label)

            self.data_on_ram_dict[index] = \
                {"image_array": image_array, "label": label}

        self.on_memory = True


class StarGanDataloader(BaseDataLoader):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 label_level=1,
                 batch_size=None,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 class_mode="binary",
                 dtype="float32"
                 ):
        self.data_getter = StarGanDataGetter(image_path_list=image_path_list,
                                             label_to_index_dict=label_to_index_dict,
                                             label_level=label_level,
                                             on_memory=on_memory,
                                             augmentation_proba=augmentation_proba,
                                             augmentation_policy_dict=augmentation_policy_dict,
                                             image_channel_dict=image_channel_dict,
                                             preprocess_input=preprocess_input,
                                             target_size=target_size,
                                             interpolation=interpolation,
                                             class_mode=class_mode,
                                             dtype=dtype
                                             )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.image_data_shape = self.data_getter[0][0]["image_array"].shape
        self.shuffle = shuffle
        self.dtype = dtype
        self.class_mode = class_mode

        self.batch_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_label_array = np.zeros(
            (self.batch_size, ), dtype=self.dtype)

        self.batch_target_image_array = np.zeros(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        self.batch_target_label_array = np.zeros(
            (self.batch_size, ), dtype=self.dtype)

        self.batch_label_array = self.data_getter.categorize_method(
            self.batch_label_array)
        self.batch_target_label_array = self.data_getter.categorize_method(
            self.batch_target_label_array)

        self.print_data_info()
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        for batch_index, total_index in enumerate(range(start, end)):
            single_data_dict, target_single_data_dict = self.data_getter[total_index]

            self.batch_image_array[batch_index] = single_data_dict["image_array"]
            self.batch_label_array[batch_index] = single_data_dict["label"]

            self.batch_target_image_array[batch_index] = target_single_data_dict["image_array"]
            self.batch_target_label_array[batch_index] = target_single_data_dict["label"]

        return np.array([self.batch_image_array, self.batch_target_image_array]), \
            np.array([self.batch_label_array, self.batch_target_label_array])

    def print_data_info(self):
        data_num = len(self.data_getter)
        print(f"Total data num {data_num} with {self.num_classes} classes")
