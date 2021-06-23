import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time

from .manage_file import remove_png_files_in_folder, get_filebasename


start = time()

batch_size = 5
train_data_length = 6200

test_predict_info = []
mode = "train"

result_path = f"images/tumor/check_state/{mode}"
os.makedirs(result_path, exist_ok=True)
remove_png_files_in_folder(result_path)

dataset_path = os.path.join("datasets", "tumor", mode)


img_path_list = [os.path.join(
    dataset_path, "image", f'{index}.png') for index in range(train_data_length)]
mask_path_list = [os.path.join(
    dataset_path, "mask", f'{index}.png') for index in range(train_data_length)]

batch_num = train_data_length // batch_size

assert train_data_length % batch_size == 0

for batch_index in range(batch_num):

    img_array_list = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                      for img_path in img_path_list[batch_index * batch_size:(batch_index + 1) * batch_size]]
    img_array_list = [cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                      for img_array in img_array_list]
    img_array_list = np.array(img_array_list)
    img_array_list = (img_array_list / 127.5) - 1
    img_mask_list = [cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                     for mask_path in mask_path_list[batch_index * batch_size:(batch_index + 1) * batch_size]]
    img_mask_list = np.array(img_mask_list)
    img_mask_list = np.round((img_mask_list / 255)).astype('float32')

    converted_img_list = generator.predict(img_array_list)
    converted_img_list = np.round(converted_img_list).astype('float32')
    predicted_img_list = converted_img_list[:, :, :, 0]

    # get predict_info
    predict_info = []
    for y_true, y_pred in zip(img_mask_list, predicted_img_list):
        predict_info.append({'f1_score': f1_score(
            np.expand_dims(y_true, axis=0), np.expand_dims(y_pred, axis=0)), 'mask_ratio': np.mean(y_true)})
    test_predict_info += predict_info

    # Rescale images 0 - 1
    img_array_list = 0.5 * img_array_list + 0.5

    row_num = 3
    column_num = 1

    for index, (img_array, predict_array, mask_array) in enumerate(zip(img_array_list, predicted_img_list, img_mask_list)):

        current_index = batch_index * batch_size + index
        current_f1_score = np.round_(
            test_predict_info[current_index]['f1_score'], 3)

        predict_array = cv2.cvtColor(predict_array, cv2.COLOR_GRAY2RGB)
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)

        gen_img = np.stack(
            [img_array, predict_array, mask_array], axis=0)

        titles = ["Original", "Model", "Man"]
        fig, axs = plt.subplots(row_num, column_num)

        for row_index in range(row_num):
            axs[row_index].imshow(gen_img[row_index])
            axs[row_index].set_title(titles[row_index])
            axs[row_index].axis("off")
        fig.set_size_inches(12, 12)
        fig.savefig(
            f"images/tumor/check_state/{mode}/{current_f1_score:.3f}_{current_index}.png")
        plt.close()

test_f1_scores = [predict_info['f1_score']
                  for predict_info in test_predict_info]

print(f'Max : {np.max(test_f1_scores)}')
print(f'Min :{np.min(test_f1_scores)}')
print(f'Mean :{np.mean(test_f1_scores)}')
print(f'Median :{np.median(test_f1_scores)}')
print(f'Elasped Time : {time()-start}')
