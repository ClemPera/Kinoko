from tensorflow.data import Dataset
from keras import KerasTensor, layers, Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, History, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_head(inputs: KerasTensor, pretrained_model_cls: KerasTensor) -> Model:
    """
    Attach a classification head on top of a backbone CLS token output.

    The head consists of a Dense(512) + ReLU, a Dropout(0.3), and a final
    sigmoid output neuron for binary classification.

    Args:
        pretrained_model_cls: The CLS token KerasTensor produced by the backbone
            (output of get_cls_dinov2()).

    Returns:
        A compiled-ready Model with a (IMG_SIZE, IMG_SIZE, 3) input shape
        and a scalar sigmoid output.
    """
    x: KerasTensor = layers.Dense(512, activation="relu")(pretrained_model_cls)
    x = layers.Dropout(0.3)(x)
    predictions: KerasTensor = layers.Dense(1, activation="sigmoid")(x)

    model: Model = Model(inputs=inputs, outputs=predictions)

    return model


def compile_model(model: Model) -> Model:
    """
    Compile a Keras model with Adam, binary crossentropy, and classification metrics.

    Args:
        model: An uncompiled Model instance.

    Returns:
        The same model, compiled in-place and returned for convenience.
    """
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall", "precision"]
    )
    return model


def train(model: Model, train_ds: Dataset, val_ds: Dataset, epochs: int) -> History:
    """
    Train a model with early stopping, LR scheduling, and checkpointing.

    Callbacks:
        - EarlyStopping: stops if val_loss does not improve for 5 epochs,
          restoring the best weights.
        - ReduceLROnPlateau: halves LR if val_loss stagnates for 3 epochs,
          down to a minimum of 1e-7.
        - ModelCheckpoint: saves the best model to checkpoints/dinov2.

    Args:
        model: A compiled Model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.

    Returns:
        A keras History object containing per-epoch metric logs.
    """
    callbacks: list[Callback] = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=f"checkpoints/dinov2.keras",
            save_best_only=True,
            save_freq="epoch",
        ),
    ]

    history: History = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return history


def load_model_from_checkpoint(checkpoint_path: str) -> Model:
    """
    Load a saved Keras model from a checkpoint file.

    Args:
        checkpoint_path: Path to a .keras checkpoint file.

    Returns:
        The restored Model.
    """
    model: Model = load_model(checkpoint_path)  # type: ignore
    return model
