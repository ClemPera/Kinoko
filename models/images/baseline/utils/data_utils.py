# ===================== IMPORTS ======================
import os

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import Dataset
# ====================================================


# ░░░░░░░░░░░░░░ 📊 Data ░░░░░░░░░░░░░░
def load_data(batch_size=32) -> tuple[Dataset, Dataset]:
    """
    Load traning and validation datasets from a directory of images.

    This function create a batched dataset for traning and validation.

    //// Architecture ////
    - Dataset directory
    - Labels : inferred binary
    - Label mode : binary
    - Validation split
    - Subset : training & validation
    - Seed : 123
    - Image size : resized to 128x128 pixels
    - Batch size : 32
    - shuffle : True for train_ds & val_ds | False for val_ds_predict


    //// Returns ////
    - Train_ds : Dataset containing the training images & labels
    - Val ds : Dataset containing the validation images & labels
    - Val ds predict : Dataset for post-training predictions and evaluation only.
                        shuffle=False guarantees a fixed order so that predictions
                        and true labels stay synchronized when compared.
                        ⚠️ Do NOT use for fit() — use val_ds instead.
    """

    dataset_dir = os.path.abspath("../data/image_dataset")

    train_ds = image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(128, 128),
        batch_size=batch_size)

    val_ds = image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=batch_size)

    val_ds_predict = image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=batch_size,
        shuffle=False)

    return train_ds, val_ds, val_ds_predict
