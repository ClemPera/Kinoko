# ===================== IMPORTS ======================
import os

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import callbacks

from model import initialize_baseline_model, compile_baseline_model, initialize_augmented_model, compile_augmented_model
# ====================================================


# ===================== 🧱 Baseline ======================
def baseline_model():
    """
    Initialize & compile the baseline model
    """

    baseline = initialize_baseline_model()
    baseline = compile_baseline_model(baseline)
    baseline.summary()
    return baseline

# ===================== 💪 Augmented ======================
def augmented_model():
    """
    Initialize & compile the augmented model
    """

    augmented = initialize_augmented_model()
    augmented = compile_augmented_model(augmented)
    augmented.summary()
    return augmented

    # ===================== 💾 Save best model ======================
def save_model():
    """
    Create a folder named 'models' and save models
    """

    os.makedirs("models", exist_ok=True)
    MODEL_1 = "models/model_1.keras"
    MODEL_2 = "models/augmented_1.keras"
    return MODEL_1, MODEL_2

    # ===================== 🏋 Training history logs ======================
def save_logs(timestamp):
    """
    Create a folder 'logs' and save history
    """

    os.makedirs("logs", exist_ok=True)
    csv_logger_1 = CSVLogger(f"logs/baseline_history_{timestamp}.csv")
    csv_logger_2 = CSVLogger(f"logs/augmented_history_{timestamp}.csv")
    return csv_logger_1, csv_logger_2

# ===================== 🔧 Callbacks ======================
def callback(MODEL_1, MODEL_2):
    """
    Create and return callbacks for each model
    """

    ES = EarlyStopping(
        patience=15,
        restore_best_weights=True
    )
    checkpoint_1 = callbacks.ModelCheckpoint(MODEL_1,
                                                monitor="val_loss",
                                                verbose=0,
                                                save_best_only=True)

    checkpoint_2 = callbacks.ModelCheckpoint(MODEL_2,
                                                monitor="val_loss",
                                                verbose=0,
                                                save_best_only=True)

    lr_reducer_1 = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.1,
                                            patience=3,
                                            verbose=1,
                                            min_lr=1e-6)

    lr_reducer_2 = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.5,
                                            patience=3,
                                            verbose=1,
                                            min_lr=1e-6)
    return ES, checkpoint_1, checkpoint_2, lr_reducer_1, lr_reducer_2


# ===================== 🏋 Training ======================
def train_model(train_ds, val_ds, timestamp, batch_size=32, epochs=50, verbose=1):
    """
    Training of the two models
    """

    # ░░ Models ░░
    baseline = baseline_model()
    augmented = augmented_model()

    # ░░ Save ░░
    MODEL_1, MODEL_2 = save_model()
    csv_logger_1, csv_logger_2 = save_logs(timestamp)

    # ░░ Callbacks ░░
    ES, checkpoint_1 ,checkpoint_2, lr_reducer_1, lr_reducer_2 = callback(MODEL_1, MODEL_2)
    callbacks_1 = [ES, lr_reducer_1, checkpoint_1, csv_logger_1]
    callbacks_2 = [ES, lr_reducer_2, checkpoint_2, csv_logger_2]

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
