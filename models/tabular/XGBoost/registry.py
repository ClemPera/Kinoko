import glob
import os
import time
from colorama import Fore, Style

import joblib
from xgboost import XGBClassifier
from keras import Model, models

import mlflow
from mlflow.tracking import MlflowClient

# Var d "environnement"

MLFLOW_TRACKING_URI='https://mlflow.lewagon.ai'
MLFLOW_EXPERIMENT='kinoko_2207'
MLFLOW_MODEL_NAME='kinoko_tab'

def save_model(model, path="ml_logic"):
    """
    Save the model with timestamp
    - works for keras or other models
    """
    # Save locally
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if isinstance(model, Model):
        model_path = f"{path}/models/{timestamp}.keras"
        model.save(model_path)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

    else:
        model_path = f"{path}/models/{timestamp}.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

    print(f"✅ Model saved at {model_path}")
    print("✅ Model saved to MLflow")

    return None


def load_model(target, path="../models/tabular/XGBoost", stage="Production"):
    """
    Load the model
    - works for different models
    """
    # If we want to load from our last locally saved model
    if target == "local":
        print(Fore.BLUE + "\nLoad latest model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(path, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + "\nLoad latest model from disk..." + Style.RESET_ALL)

        if most_recent_model_path_on_disk.endswith(".keras") or most_recent_model_path_on_disk.endswith(".h5"):
            model = models.load_model(most_recent_model_path_on_disk)

        elif most_recent_model_path_on_disk.endswith(".json"):
            model = XGBClassifier()
            model.load_model(most_recent_model_path_on_disk)

        else:  # .pkl
            model = joblib.load(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return model

    # If we want to load from mlflow

    elif target == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None



def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper
