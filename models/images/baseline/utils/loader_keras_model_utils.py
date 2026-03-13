# ===================== IMPORTS ======================
from pathlib import Path
from tensorflow.keras.models import load_model

# ====================================================


def load_keras_model(model_file: str):
    """
    Load Keras model from 'DL_logic/models/' for an API endpoint

    //// Return ////
    Charged model
    """

    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Le modèle {model_file} n'existe pas dans {model_path}")

    return load_model(model_path)
