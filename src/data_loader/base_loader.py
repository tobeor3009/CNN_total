import tensorflow


class BaseDataGetter():

    def __len__(self):
        return len(self.image_path_list)    

class BaseDataLoader(tensorflow.keras.utils.Sequence):

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()
  