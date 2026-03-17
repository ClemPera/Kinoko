from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE, Dataset

SEED: int = 42


def train_test_split(image_dir: str,
                     image_sizes=(518, 518),
                     batch_size=32) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split an image directory into train, validation, and test datasets.

    The split ratio is 70% train, 15% validation, 15% test.
    Datasets are prefetched for optimal GPU throughput.

    Args:
        image_dir: Path to the directory containing labelled image subdirectories.

    Returns:
        A tuple of (train_ds, test_ds, val_ds) as prefetched tf.data.Dataset objects.
    """

    train_ds: Dataset = image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.3,
        subset="training",
        seed=SEED,
        image_size=image_sizes,
        batch_size=batch_size
    )

    test_val_ds: Dataset = image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.3,
        subset="validation",
        seed=SEED,
        image_size=image_sizes,
        batch_size=batch_size
    )

    half_test_val_size: int = int(len(test_val_ds) / 2)
    test_ds: Dataset = test_val_ds.take(half_test_val_size)
    val_ds: Dataset = test_val_ds.skip(half_test_val_size)

    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return (train_ds, test_ds, val_ds)
