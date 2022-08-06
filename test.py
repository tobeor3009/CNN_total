from time import sleep
import multiprocessing as mp
from multiprocessing.queues import Empty
import math
import random
import numpy as np

WAIT_TIME = 0.05


def default_collate_fn(data_object_list):

    batch_image_array = []
    batch_label_array = []
    for image_array, label_array in data_object_list:
        batch_image_array.append(image_array)
        batch_label_array.append(label_array)
    batch_image_array = np.stack(batch_image_array, axis=0)
    batch_image_array = np.stack(batch_label_array, axis=0)

    return batch_image_array, batch_label_array


def consumer_fn(data_getter, idx_queue, output_queue):
    inter_idx = 0
    while True:
        inter_idx += 1
        try:
            idx = idx_queue.get(timeout=0)
        except Empty:
            sleep(WAIT_TIME)
            continue
        if idx is None:
            break
        data = (idx, data_getter[idx])
        output_queue.put(data)


class DataPool():
    def __init__(self, data_getter, batch_size, num_workers,
                 collate_fn=default_collate_fn, shuffle=False):
        self.data_getter = data_getter
        self.collate_fn = collate_fn
        self.data_num = len(data_getter)
        self.idx_list = list(range(self.data_num))
        if shuffle:
            self.shuffle_idx()
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.batch_size = batch_size

        self.batch_idx = 0
        self.batch_idx_list = None

        self.idx_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.process_list = []
        for _ in range(num_workers):
            process = mp.Process(target=consumer_fn, args=(self.data_getter,
                                                           self.idx_queue,
                                                           self.output_queue))
            process.daemon = True
            process.start()
            self.process_list.append(process)

    def __iter__(self):
        self.reset_states()
        return self

    def __next__(self):
        if self.batch_idx == self.batch_num:
            self.reset_states()
            raise StopIteration
        start_idx = self.batch_size * self.batch_idx
        end_idx = min(start_idx + self.batch_size, self.data_num)
        current_batch_size = end_idx - start_idx
        batch_idx_list = self.idx_list[start_idx: end_idx]
        for batch_idx in batch_idx_list:
            self.idx_queue.put(batch_idx)

        data_object_list = []
        for idx in range(current_batch_size):
            while True:
                try:
                    batch_idx, data_object = self.output_queue.get(timeout=0)
                    break
                except Empty:
                    sleep(WAIT_TIME)
                    continue
            data_object_list.append(data_object)

        batch_data_tuple = self.collate_fn(data_object_list)

        self.batch_idx_list = batch_idx_list
        self.batch_idx += 1
        return batch_data_tuple

    def reset_states(self):
        self.batch_idx = 0
        self.batch_idx_list = None

    def shuffle_idx(self):
        random.shuffle(self.idx_list)
