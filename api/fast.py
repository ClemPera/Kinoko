from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from PIL import Image, ImageFile
from datetime import datetime

from xgboost import XGBClassifier
from keras import Model

from models.tabular.XGBoost.registry import load_model as XGBoost_load_model
from models.tabular.XGBoost.model import predict as XGBoost_predict
from models.images.dinov2.model import load_model_from_checkpoint as dinov2_load_model_
from models.images.dinov2.inference import predict as dinov2_predict
from models.images.baseline.utils import load_keras_model as baseline_load_model
from models.images.baseline.evaluate import prediction as baseline_prediction
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
    cap_shape: str | None = None,
    cap_color: str | None = None,
    does_bruise_or_bleed: str | None = None,
    gill_attachment: str | None = None,
    gill_color: str | None = None,
    stem_color: str | None = None,
    has_ring: str | None = None,
    habitat: str | None = None,
    season: str | None = None
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
        model_XGBoost = XGBoost_load_model(
            "models/tabular/XGBoost")  # type: ignore
        assert model_XGBoost is not None

    is_poisonous, probability = XGBoost_predict(model_XGBoost, data)

    return {
        "probability": probability,
        "poisonous": is_poisonous
    }


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
                model_dinov2 = dinov2_load_model_(
                    "models/images/dinov2/checkpoints/dinov2.keras")
                assert model_dinov2 is not None

            is_poisonous, probability = dinov2_predict(model_dinov2, image)

        case "baseline":
            if model_baseline is None:
                model_baseline = baseline_load_model("model_1.keras")
                assert model_baseline is not None

            df_image = pil_to_dataset(image)
            is_poisonous, probability = baseline_prediction(
                model_baseline, df_image)

        case "baseline_aug":
            if model_baseline_aug is None:
                model_baseline_aug = baseline_load_model("augmented_1.keras")
                assert model_baseline_aug is not None

            df_image = pil_to_dataset(image)
            is_poisonous, probability = baseline_prediction(
                model_baseline_aug, df_image)

        case _:
            raise HTTPException(
                status_code=400, detail="The model selected doesn't exist")

    return {
        "probability": probability,
        "poisonous": is_poisonous
    }


@app.get("/models")
def models():
    """
    Returns the list of models available
    """
    return {
        "img_models": ["dinov2_baseline", "baseline", "baseline_aug"],
        "tab_models": ["tabular"]
    }


@app.post("/predict_all")
def predict_all(
    file: UploadFile,
    cap_shape: str | None = None,
    cap_color: str | None = None,
    does_bruise_or_bleed: str | None = None,
    gill_attachment: str | None = None,
    gill_color: str | None = None,
    stem_color: str | None = None,
    has_ring: str | None = None,
    habitat: str | None = None,
    season: str | None = None,
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
