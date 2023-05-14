# CNN Total Keras

This repository contains a collection of various libraries and utility functions for deep learning with Keras. and This is a module being used for medical tasks.
  - task
    - Data Loader
      - multi process
      - support augmentation
        - use albumentation
    - classification
    - segmentation
    - generation
      - attgan
      - stargan
      - cyclegan
      - wgan
    - 2d_3d
  - backbone
    - inception_v3
    - inception_resnet_v2
    - deeplab_v3_plus
    - swin_unet
    - vit
  - util
    - pathology
      - vahadane
        - seprated h and e
        - stain normalization
      - xray_ct
        - ct_resizing
        - fastmarching filter

## Installation Or How to use it.

To be Done. I create custom .ipynb files for each task to use this module, and I haven't uploaded any files yet because the general usage method hasn't been confirmed. If there is a request, I will upload examples of the .ipynb files.
But The files under the 'src/model' directory should be easy to use as they return tensorflow.keras.Model objects. If needed, you can directly import and use them in your projects.

## Why should you use this repository?
If you become proficient in handling this code, you can quickly carry out various tasks or experiments just by adjusting parameters.
