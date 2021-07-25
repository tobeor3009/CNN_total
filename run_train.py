import json

# parse config.json
with open("./config.json") as binary_json:
    config_dict = json.load(binary_json)

task = str(config_dict["task"])

if task == "classification":
  from src.manage_train import classification  
elif task == "segmentation":
  from src.manage_train import segmentation
# elif task == "cycle_gan":
#   from src.manage_train import cycle_gan
