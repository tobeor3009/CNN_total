import os
import json
import shutil
from glob import glob
from datetime import date
from ast import literal_eval as make_tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Nadam

import segmentation_models as sm
sm.set_framework ('tf.keras')

from src.data_loader.segmentation import SegDataloader

# parse config.json
with open("./config.json") as binary_json:
    config_dict = json.load(binary_json)

gpu_number = str(config_dict["gpu_number"])
task = str(config_dict["task"])
data_set_name = str(config_dict["data_set_name"])
batch_size = int(config_dict["batch_size"])
on_memory = bool(config_dict["on_memory"])
argumentation_proba = float(config_dict["argumentation_proba"])
target_size = make_tuple(config_dict["target_size"])
interpolation = str(config_dict["interpolation"])
class_mode = str(config_dict["class_mode"])
dtype = str(config_dict["dtype"])

# set data_path_config
common_data_path = f"./datasets/{task}/{data_set_name}/"
train_image_path_regexp = f"{common_data_path}/{config_dict['train_image_path_regexp']}"
train_mask_path_regexp = f"{common_data_path}/{config_dict['train_mask_path_regexp']}"
valid_image_path_regexp = f"{common_data_path}/{config_dict['valid_image_path_regexp']}"
valid_mask_path_regexp = f"{common_data_path}/{config_dict['valid_mask_path_regexp']}"
test_image_path_regexp = f"{common_data_path}/{config_dict['test_image_path_regexp']}"
test_mask_path_regexp = f"{common_data_path}/{config_dict['test_mask_path_regexp']}"

# define model config
model_backbone = str(config_dict["model_backbone"])
last_channel_activation = str(config_dict["last_channel_activation"])

# define train config
learning_rate = str(config_dict["learning_rate"])
weight_save_format = str(config_dict["weight_save_format"])
code_test_by_small_data = bool(config_dict["code_test_by_small_data"])
small_data_num = int(config_dict["small_data_num"])

if gpu_number == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gpu_devices = tf.config.experimental.list_physical_devices("CPU")
else:
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

################ Define Data Loader ################

train_image_path_list = glob(train_image_path_regexp)
train_mask_path_list = glob(train_mask_path_regexp)
valid_image_path_list = glob(valid_image_path_regexp)
valid_mask_path_list = glob(valid_mask_path_regexp)
test_image_path_list = glob(test_image_path_regexp)
test_mask_path_list = glob(test_mask_path_regexp)

if code_test_by_small_data:
  train_image_path_list = train_image_path_list[:small_data_num]
  train_mask_path_list = train_mask_path_list[:small_data_num]
  valid_image_path_list = valid_image_path_list[:small_data_num]
  valid_mask_path_list = valid_mask_path_list[:small_data_num]
  test_image_path_list = test_image_path_list[:small_data_num]
  test_mask_path_list = test_mask_path_list[:small_data_num]

train_data_loader = SegDataloader(image_path_list=train_image_path_list,
                                  mask_path_list=train_mask_path_list,
                                  batch_size=batch_size,
                                  on_memory=on_memory,
                                  argumentation_proba=argumentation_proba,
                                  preprocess_input=preprocess_input,
                                  target_size=target_size,
                                  interpolation=interpolation,
                                  shuffle=True,
                                  dtype=dtype
                                  )
valid_data_loader = SegDataloader(image_path_list=valid_image_path_list,
                                  mask_path_list=valid_mask_path_list,
                                  batch_size=batch_size,
                                  on_memory=on_memory,
                                  argumentation_proba=0,
                                  preprocess_input=preprocess_input,
                                  target_size=target_size,
                                  interpolation=interpolation,
                                  shuffle=True,
                                  dtype=dtype
                                  )
test_data_loader = SegDataloader(image_path_list=test_image_path_list,
                                 mask_path_list=test_mask_path_list,
                                 batch_size=1,
                                 on_memory=False,
                                 argumentation_proba=0,
                                 preprocess_input=preprocess_input,
                                 target_size=target_size,
                                 interpolation=interpolation,
                                 shuffle=False,
                                 dtype=dtype
                                 )

################ Define and Compile Model ################

# create the base pre-trained model~
model = sm.Unet(backbone_name=model_backbone, 
                input_shape=(None, None, 3), 
                classes=1, 
                activation='sigmoid')

################ Define Callbacks ################

today = date.today()
# YY/MM/dd
today_str = today.strftime("%Y-%m-%d")
today_weight_path = f"./result/{task}/{data_set_name}/{today_str}/{gpu_number}/target_size_{target_size}/weights/" 
today_logs_path = f"./result/{task}/{data_set_name}/{today_str}/{gpu_number}/target_size_{target_size}/"
os.makedirs(today_weight_path, exist_ok=True)
os.makedirs(today_logs_path, exist_ok=True)
shutil.copy("./config.json", f"{today_logs_path}/config.json")

checkpoint_callback = ModelCheckpoint(
    today_weight_path+"/weights_{val_loss:.4f}_{loss:.4f}_{epoch:02d}.hdf5",
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='min')

reduceLROnPlat_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=20,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=5,
    min_lr=1e-7)

csv_logger_callback = CSVLogger(f'{today_logs_path}/log.csv', append=False, separator=',')

################ Run train ################

start_epoch = 0
epochs = 200

model.fit(
    train_data_loader,
    validation_data=valid_data_loader,
    epochs=epochs,
    callbacks=[checkpoint_callback,
              reduceLROnPlat_callback,
              csv_logger_callback],
    initial_epoch=start_epoch
)

################ Evaluate test Dataset ################

