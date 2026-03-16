import glob
import os
import time

import joblib
from xgboost import XGBClassifier
from keras import Model, models
from sklearn.pipeline import Pipeline


def save_model(model, path="ml_logic"):
    """
    Save the model with timestamp

    Args:
        - model: works for keras or other models
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if isinstance(model, Model):
        model_path = f"{path}/models/{timestamp}.keras"
        model.save(model_path)

    else:
        model_path = f"{path}/models/{timestamp}.pkl"
        joblib.dump(model, model_path)

    print(f"✅ Model saved at {model_path}")


def load_model(path="ml_logic"):
    """
    Load the model
    - works for different models
    args: 
        - path: model folder path
    """
    print("\nLoad latest model from local registry...")

    local_model_directory = os.path.join(path, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print("\nLoad latest model from disk...")

    if most_recent_model_path_on_disk.endswith(".keras") or most_recent_model_path_on_disk.endswith(".h5"):
        model = models.load_model(most_recent_model_path_on_disk)

    elif most_recent_model_path_on_disk.endswith(".json"):
        model = XGBClassifier()
        model.load_model(most_recent_model_path_on_disk)

    else:  # .pkl
        model = joblib.load(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return model
