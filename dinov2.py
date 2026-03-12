import os

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE, Dataset

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import keras_hub

import numpy as np

IMG_SIZE: int = 518
BATCH_SIZE: int = 32
EPOCHS: int = 50
SEED: int = 42


def train_test_split(image_dir: str) -> tuple[Dataset, Dataset, Dataset]:
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
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    test_val_ds: Dataset = image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.3,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    half_test_val_size: int = int(len(test_val_ds) / 2)
    test_ds: Dataset = test_val_ds.take(half_test_val_size)
    val_ds: Dataset = test_val_ds.skip(half_test_val_size)

    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return (train_ds, test_ds, val_ds)


def get_cls_dinov2() -> keras.KerasTensor:
    """
    Build a frozen DINOv2 backbone and return its CLS token output.

    The backbone is loaded from the "dinov2_base" preset and frozen.
    Input images are rescaled to [0, 1] and normalized with ImageNet statistics
    before being passed to the backbone.

    Returns:
        - A KerasTensor of shape (batch, hidden_dim) representing the CLS token
            extracted from the last layer of the backbone.
        - A Kera Input reprensenting the inputs used by the model
    """
    backbone: keras_hub.models.DINOV2Backbone = keras_hub.models.DINOV2Backbone.from_preset("dinov2_base")
    backbone.trainable = False

    inputs: keras.KerasTensor = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x: keras.KerasTensor = keras.layers.Rescaling(1 / 255.0)(inputs)
    x = keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    )(x)

    backbone_out: keras.KerasTensor = backbone({"images": x})
    outputs: keras.KerasTensor = backbone_out[:, 0, :]

    return inputs, outputs


def create_head(inputs: keras.Input, pretrained_model_cls: keras.KerasTensor) -> keras.Model:
    """
    Attach a classification head on top of a backbone CLS token output.

    The head consists of a Dense(512) + ReLU, a Dropout(0.3), and a final
    sigmoid output neuron for binary classification.

    Args:
        pretrained_model_cls: The CLS token KerasTensor produced by the backbone
            (output of get_cls_dinov2()).

    Returns:
        A compiled-ready keras.Model with a (IMG_SIZE, IMG_SIZE, 3) input shape
        and a scalar sigmoid output.
    """
    x: keras.KerasTensor = keras.layers.Dense(512, activation="relu")(pretrained_model_cls)
    x = keras.layers.Dropout(0.3)(x)
    predictions: keras.KerasTensor = keras.layers.Dense(1, activation="sigmoid")(x)

    model: keras.Model = keras.Model(inputs=inputs, outputs=predictions)

    return model


def compile_model(model: keras.Model) -> keras.Model:
    """
    Compile a Keras model with Adam, binary crossentropy, and classification metrics.

    Args:
        model: An uncompiled keras.Model instance.

    Returns:
        The same model, compiled in-place and returned for convenience.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall", "precision"]
    )
    return model


def train(model: keras.Model, train_ds: Dataset, val_ds: Dataset) -> keras.callbacks.History:
    """
    Train a model with early stopping, LR scheduling, and checkpointing.

    Callbacks:
        - EarlyStopping: stops if val_loss does not improve for 5 epochs,
          restoring the best weights.
        - ReduceLROnPlateau: halves LR if val_loss stagnates for 3 epochs,
          down to a minimum of 1e-7.
        - ModelCheckpoint: saves the best model to checkpoints/dinov2.keras.

    Args:
        model: A compiled keras.Model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.

    Returns:
        A keras History object containing per-epoch metric logs.
    """
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"checkpoints/dinov2.keras",
            save_best_only=True,
            save_freq="epoch",
        ),
    ]

    history: keras.callbacks.History = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return history


def load_model_from_checkpoint(checkpoint_path: str) -> keras.Model:
    """
    Load a saved Keras model from a checkpoint file.

    Args:
        checkpoint_path: Path to a .keras checkpoint file.

    Returns:
        The restored keras.Model.
    """
    model: keras.Model = keras.models.load_model(checkpoint_path)
    return model


def test(model: keras.Model, test_ds: Dataset) -> list[float]:
    """
    Evaluate a model on the test dataset and print the results.

    Args:
        model: A trained and compiled keras.Model.
        test_ds: Test tf.data.Dataset.

    Returns:
        A list of scalar metric values: [loss, accuracy, recall, precision].
    """
    results: list[float] = model.evaluate(test_ds)
    print(f"loss: {results[0]}\naccuracy: {results[1]}\nrecall: {results[2]}\nprecision: {results[3]}")
    return results

def predict(model: keras.Model, image_path: str) -> tuple[str, float]:
    """
    Run inference on a single image and return the predicted class and confidence.

    The image is loaded, resized to (IMG_SIZE, IMG_SIZE), and passed through
    the model. The output is a sigmoid probability interpreted as:
        >= 0.5 → class 1
        <  0.5 → class 0

    Args:
        model: A trained keras.Model.
        image_path: Path to the image file to classify.

    Returns:
        A tuple of (predicted_class, confidence) where predicted_class is a
        string ("0" or "1") and confidence is the raw sigmoid output in [0, 1].
    """
    img: keras.KerasTensor = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array: np.ndarray = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)

    confidence: float = float(model.predict(img_array)[0][0])
    predicted_class: str = "1" if confidence >= 0.5 else "0"

    return predicted_class, confidence
