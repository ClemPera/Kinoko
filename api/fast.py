from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from PIL import Image

from models.tabular import XGBoost
from models.images.dinov2 import dinov2
from keras import Model

app = FastAPI()

# Image models
model_dinov2: Model | None = None
model_baseline: Model | None = None

# Tabular models
model_XGBoost: Model | None = None

# Allowing all middleware is optional, but good practice for dev purposes
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
    global model_XGBoost

    data = pd.DataFrame([
        cap_shape,
        cap_color,
        does_bruise_or_bleed,
        gill_attachment,
        gill_color,
        stem_color,
        has_ring,
        habitat,
        season
    ])

    if not model_XGBoost:
        # TODO
        model_XGBoost = XGBoost.model.load("model/tabular/checkpoints/XGBoost.keras")
    
    result = XGBoost.model.predict(model_XGBoost, data)

    return {
        "result": result
    }

@app.post("/predict_img")
def predict_img(
    model: str,
    file: UploadFile,
):
    global model_dinov2, model_baseline

    image: Image.ImageFile.ImageFile = Image.open(file.file)
    match model:
        case "dinov2_baseline":
            if not model_dinov2:
                model_dinov2 = dinov2.load_model_from_checkpoint("models/images/dinov2/checkpoints/dinov2.keras")
            
            predicted_class, confidence = dinov2.predict(model_dinov2, image)
        case "baseline":
            # TODO
            if not model_baseline:
                pass
                # model_baseline = 
            pass

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

@app.get("/models")
def models():
    return {
        "img_models": ["dinov2_baseline", "baseline"],
        "tab_models": ["XGboost"]
    }

@app.get("/predict_all")
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

