import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback

class SegmentationMonitor(Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, data_loader, save_image_path, num_img=4, reverse_process=None):
        self.data_loader = data_loader
        self.num_img = num_img
        self.save_image_path = save_image_path
        self.class_num = self.model.output.shape[-1]
        if reverse_process is None:
            self.reverse_process = lambda x: ((x + 1) * 127.5).astype(np.uint8)
        else:
            self.reverse_process = reverse_process
        os.makedirs(save_image_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):

        len_data_loader = len(self.data_loader.data_getter)
        random_index = range(len_data_loader)
        random_index = np.random.permutation(random_index)[:self.num_img]

        _, ax = plt.subplots(self.num_img, 3, figsize=(self.num_img * 6, 12))
        for current_index, image_index in enumerate(random_index):
            image_array, mask_array = self.data_loader.data_getter[image_index].values(
            )

            image_array = np.expand_dims(image_array, axis=0)
            prediction = self.model(image_array).numpy()[0, ..., 0]
            image_array = self.reverse_process(image_array[0])

            ax[current_index, 0].imshow(image_array)
            ax[current_index, 0].set_title(f"IMAGE")
            ax[current_index, 1].imshow(mask_array, cmap="gray")
            ax[current_index, 1].set_title(f"GT")
            ax[current_index, 2].imshow(prediction, cmap="gray")
            ax[current_index, 2].set_title(f"PRED")

        plt.savefig(f"{self.save_image_path}/generated_image_{epoch+1}")
        plt.show()
        plt.close()
