# ===================== IMPORTS ======================
import os

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import callbacks

from .model import initialize_baseline_model, compile_model, initialize_augmented_model
# ====================================================


# ===================== 🧱 Baseline ======================
def baseline_model():
    """
    Initialize & compile the baseline model
    """

    baseline = initialize_baseline_model()
    baseline = compile_model(baseline)
    baseline.summary()
    return baseline

# ===================== 💪 Augmented ======================


def augmented_model():
    """
    Initialize & compile the augmented model
    """

    augmented = initialize_augmented_model()
    augmented = compile_model(augmented)
    augmented.summary()
    return augmented

    # ===================== 💾 Save best model ======================


def save_model():
    """
    Create a folder named 'models' and save models

    Returns:
        Tuple of models path (model_1, model_2)
    """

    os.makedirs("DL_logic/models", exist_ok=True)
    model_1 = "DL_logic/models/model_1.keras"
    model_2 = "DL_logic/models/augmented_1.keras"
    return model_1, model_2

    # ===================== 🏋 Training history logs ======================


def save_logs(timestamp):
    """
    Create a folder 'logs' and save history

    Args: 
        - timestamp: timestamp to add to the model file name
    Returns: 
        Tuple of CSVLogger objects
    """

    os.makedirs("DL_logic/logs", exist_ok=True)
    csv_logger_1 = CSVLogger(f"DL_logic/logs/baseline_history_{timestamp}.csv")
    csv_logger_2 = CSVLogger(
        f"DL_logic/logs/augmented_history_{timestamp}.csv")
    return csv_logger_1, csv_logger_2

    # ===================== 🔧 Callbacks ======================


def callback(model):
    """
    Create and return callbacks for each model

    Args:
        - model: model to apply callbacks to

    Returns: Tuple of all callbacks: (es, checkpoint, lr_reducer)
    """

    es = EarlyStopping(
        patience=15,
        restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(model,
                                           monitor="val_loss",
                                           verbose=0,
                                           save_best_only=True)

    lr_reducer = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                             factor=0.1,
                                             patience=3,
                                             verbose=1,
                                             min_lr=1e-6)

    return es, checkpoint, lr_reducer


# ===================== 🏋 Training ======================
def train_model(train_ds, val_ds, timestamp, batch_size=32, epochs=50, verbose=1):
    """
    Training of the two models

    Args:
        - train_ds: Train dataset
        - val_ds: Val dataset
        - timestamp: Timestamp add to files name
        - batch size: Batch size to train model on
        - epochs: Max epochs to train model on
        - verbose: Verbosity of training
    Returns: 
        - tuple with baseline history, augmented history, baseline model and augmented model
    """

    # ░░ Models ░░
    baseline = baseline_model()
    augmented = augmented_model()

    # ░░ Save ░░
    model_1, model_2 = save_model()
    csv_logger_1, csv_logger_2 = save_logs(timestamp)

    # ░░ Callbacks ░░
    es1, checkpoint_1, lr_reducer_1 = callback(model_1)
    es2, checkpoint_2, lr_reducer_2 = callback(model_2)
    callbacks_1 = [es1, lr_reducer_1, checkpoint_1, csv_logger_1]
    callbacks_2 = [es2, lr_reducer_2, checkpoint_2, csv_logger_2]

    # ░░ Baseline training ░░
    history_1 = baseline.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_ds,
        verbose=verbose,
        callbacks=callbacks_1
    )

    # ░░ Augmented training ░░
    history_2 = augmented.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_ds,
        verbose=verbose,
        callbacks=callbacks_2
    )

    return history_1, history_2, baseline, augmented
