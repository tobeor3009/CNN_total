import math
import numpy as np
from PIL import Image


def get_image_patch_num_2d(image_shape, patch_size, stride):
    image_row, image_col = image_shape
    patch_row, patch_col = patch_size
    stride_row, stride_col = stride

    patch_row_num = math.ceil((image_row - patch_row) / stride_row)
    patch_row_num += 2 if patch_row_num % 2 == 0 else 1
    patch_col_num = math.ceil((image_col - patch_col) / stride_col)
    patch_col_num += 2 if patch_col_num % 2 == 0 else 1

    return patch_row_num, patch_col_num


def get_image_patch_num_3d(image_shape, patch_size, stride):
    image_z, image_row, image_col = image_shape
    patch_z, patch_row, patch_col = patch_size
    stride_z, stride_row, stride_col = stride

    patch_z_num = math.ceil((image_z - patch_z) / stride_z)
    patch_z_num += 2 if patch_z_num % 2 == 0 else 1
    patch_row_num = math.ceil((image_row - patch_row) / stride_row)
    patch_row_num += 2 if patch_row_num % 2 == 0 else 1
    patch_col_num = math.ceil((image_col - patch_col) / stride_col)
    patch_col_num += 2 if patch_col_num % 2 == 0 else 1

    return patch_z_num, patch_row_num, patch_col_num


def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.

    Input
    ----------
        The file path of the stored keras model.

    Output
    ----------
        Weights of the model.

    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W


def image_to_array(filenames, size, channel):
    '''
    Converting RGB images to numpy arrays.

    Input
    ----------
        filenames: an iterable of the path of image files
        size: the output size (height == width) of image. 
              Processed through PIL.Image.NEAREST
        channel: number of image channels, e.g. channel=3 for RGB.

    Output
    ----------
        An array with shape = (filenum, size, size, channel)

    '''

    # number of files
    L = len(filenames)

    # allocation
    out = np.empty((L, size, size, channel))

    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ..., 0] = np.array(pix)
    else:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]


def shuffle_ind(L):
    '''
    Generating random shuffled indices.

    Input
    ----------
        L: an int that defines the largest index

    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''

    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind


def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model

    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model
