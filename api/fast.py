from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from PIL import Image, ImageFile
from datetime import datetime

from xgboost import XGBClassifier
from keras import Model

from models.tabular import XGBoost
from models.images.dinov2 import dinov2
from models.images import baseline
from .utils import *

app = FastAPI()

# Image models
model_dinov2: Model | None = None
model_baseline: Model | None = None
model_baseline_aug: Model | None = None

# Tabular models
model_XGBoost: XGBClassifier | None = None

# Allowing all middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict_tab")
def predict_tab(
    cap_shape: str,
    cap_color: str,
    does_bruise_or_bleed: str,
    gill_attachment: str,
    gill_color: str,
    stem_color: str,
    has_ring: str,
    habitat: str,
    season: str
):
    """
    Runs the predict on the tabular model with tabular inputs

    Args: 
        All mushrooms features handled

    Returns:
        A list containing a key `results` containing predicted `label` and `prob`
    """
    global model_XGBoost

    data = pd.DataFrame([{
        "cap_shape": cap_shape,
        "cap_color": cap_color,
        "does_bruise_or_bleed": does_bruise_or_bleed,
        "gill_attachment": gill_attachment,
        "gill_color": gill_color,
        "stem_color": stem_color,
        "has_ring": has_ring,
        "habitat": habitat,
        "season": season
    }])

    if model_XGBoost is None:
        model_XGBoost = XGBoost.registry.load_model("models/tabular/XGBoost")
        assert model_XGBoost != None

    result = XGBoost.model.predict(model_XGBoost, data)

    return result


@app.post("/predict_img")
def predict_img(
    model: str,
    file: UploadFile,
):
    """
    Runs the predict on the image model 

    Args: 
        model: string of the model to use. Please see endpoint `/models`
        file: image to predict on

    Returns:
        A json containing `probability` and `class`
    }
    """
    global model_dinov2, model_baseline, model_baseline_aug

    image: ImageFile.ImageFile = Image.open(file.file)
    match model:
        case "dinov2_baseline":
            if model_dinov2 is None:
                model_dinov2 = dinov2.load_model_from_checkpoint(
                    "models/images/dinov2/checkpoints/dinov2.keras")
                assert model_dinov2 != None

            class_names, probability = dinov2.predict(model_dinov2, image)

        case "baseline":
            if model_baseline is None:
                model_baseline = baseline.utils.load_keras_model(
                    "model_1.keras")
                model_baseline_aug = baseline.utils.load_keras_model(
                    "augmented_1.keras")
                assert model_baseline != None
                assert model_baseline_aug != None

            df_image = pil_to_dataset(image)
            df_proba_results = baseline.evaluate.prediction(
                model_baseline,
                model_baseline_aug,
                df_image,
                ['edible', 'poisonous'],
                datetime.now().strftime("%d%m%Y_%H%M%S")
            )

            probability: dict = {"Baseline": str(df_proba_results["Baseline_proba"]),
                                 "Augmented": str(df_proba_results["Augmented_proba"])}

            class_names: dict = {"Baseline": str(df_proba_results["Baseline_class"]),
                                 "Augmented:": str(df_proba_results["Augmented_class"])}
        case _:
            raise HTTPException(
                status_code=400, detail="The model selected doesn't exist")

    return {
        "probability": probability,
        "class": class_names
    }


@app.get("/models")
def models():
    """
    Returns the list of models available
    """
    return {
        "img_models": ["dinov2_baseline", "baseline"],
        "tab_models": ["XGboost"]
    }


@app.post("/predict_all")
def predict_all(
    cap_shape: str,
    cap_color: str,
    does_bruise_or_bleed: str,
    gill_attachment: str,
    gill_color: str,
    stem_color: str,
    has_ring: str,
    habitat: str,
    season: str,
    file: UploadFile,
):
    """
    Run the predictions for every models available

    Args:
        All mushrooms features handled
        file: image to predict on
    Returns:
        All models and their prediction results
    """
    results: dict[str, dict] = {
        "img_models": {},
        "tab_models": {}
    }

    all_models = models()

    for model in all_models["img_models"]:
        results["img_models"][model] = predict_img(model, file)

    for model in all_models["tab_models"]:
        results["tab_models"][model] = predict_tab(cap_shape,
                                                   cap_color,
                                                   does_bruise_or_bleed,
                                                   gill_attachment,
                                                   gill_color,
                                                   stem_color,
                                                   has_ring,
                                                   habitat,
                                                   season,
                                                   )

    return results
