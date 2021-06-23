import os

datasets_path = "./datasets/"
weights_path = "./weights/"
logs_path = "./logs/"

os.makedirs(f'{datasets_path}/classification', exist_ok=True)
os.makedirs(f'{datasets_path}/segmentation', exist_ok=True)

os.makedirs(f'{weights_path}/classification', exist_ok=True)
os.makedirs(f'{weights_path}/segmentation', exist_ok=True)

os.makedirs(f'{logs_path}/classification', exist_ok=True)
os.makedirs(f'{logs_path}/segmentation', exist_ok=True)
