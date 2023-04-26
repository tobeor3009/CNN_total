import cv2
import numpy as np
from skimage.morphology import binary_closing, remove_small_holes, binary_erosion, binary_dilation, remove_small_objects

def segment_background_and_tissue(image_array_3d, axis_first=True):
    
    if axis_first:
        z_num = image_array_3d.shape[0]
        h, w = image_array_3d.shape[1:]
    else:
        z_num = image_array_3d.shape[-1]
        h, w = image_array_3d.shape[:-1]
    # Load NIfTI image data
    image_array_3d = (image_array_3d - image_array_3d.min()) / (image_array_3d.max() - image_array_3d.min())
    image_array_3d = np.round(image_array_3d * 255).astype("uint8")
    # Threshold image using Otsu's method
    brain_mask = []
    for idx in range(z_num):
        # Compute background mask
        if axis_first:
            image_slice = image_array_3d[idx, ...]
        else:
            image_slice = image_array_3d[..., idx]
        _, thresh = cv2.threshold(image_slice, 0, 255, cv2.THRESH_OTSU)
        brain_mask_slice = image_slice < thresh
        for idx in range(3):
            brain_mask_slice = binary_erosion(brain_mask_slice)
        brain_mask_slice = remove_small_objects(brain_mask_slice, min_size=h * w * 0.1)
        for idx in range(3):
            brain_mask_slice = binary_dilation(brain_mask_slice)
        
        brain_mask.append(brain_mask_slice)
    if axis_first:
        brain_mask = np.stack(brain_mask, axis=0)
    else:
        brain_mask = np.stack(brain_mask, axis=-1)
    for _ in range(5):
        brain_mask = binary_closing(brain_mask)
    brain_mask = remove_small_holes(brain_mask, area_threshold=h * w * z_num * 0.25) 
    background_mask = np.invert(brain_mask)

    return background_mask, brain_mask