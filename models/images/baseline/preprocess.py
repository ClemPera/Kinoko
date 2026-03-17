# ===================== IMPORTS ======================
from pathlib import Path

import numpy as np

from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array

# =====================================================


#  ░░░░░░░░░░░░░░ DATA AUGMENTATION ░░░░░░░░░░░░░░

def augment_edible(edible_dir="data/image_dataset/edible"):
    """
    Simple data augmentation to edible folder by creating a horizontally flipped
    of each image.

    Augmented images are saved alongside the original ones.
    The new file name is prefixed with 'aug_'.

    Running this function will create new files in the dataset.
    """

    paths = list(Path(edible_dir).rglob("*.png"))

    for p in paths:
        save_img(str(p.parent / f"aug_{p.name}"),
                 np.fliplr(img_to_array(load_img(p))))
