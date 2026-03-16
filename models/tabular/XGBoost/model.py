import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

from .data import get_data, get_data_reduced
from .preprocess import preprocess_features, tts
from .utils import add_noise_to_dataset

def define_model():
    """
    Define the model to perform on tabular data
    """
    model = XGBClassifier(objective="binary:logistic",
                          eval_metric="logloss",
                          random_state=3)
    return model

def train_model(X_train,
                y_train,
                pipeline):
    """
    Training the model:
    - X_train, y_train : from tts()
    - pipeline: from preprocess_features()
    """
    # call the function made before
    model = define_model()

    # Pipeline with data preprocessed and model
    pipe_model = make_pipeline(pipeline, model)

    # train model
    pipe_model.fit(X_train, y_train)

    return pipe_model


def predict(model, data) -> tuple[bool, float]:
    """
    Prediction function, gives out the predicted class and the associated prob
    - model : model trained before
    - data : can be a single data to predict
    """
    #Predict and prob
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)

    return bool(pred), float(proba[0][pred])

def predict_multiple(model, data):
    """
    Prediction function, gives out the predicted class and the associated prob
    - model : model trained before
    - data : can be a single data to predict, or a dataframe with xx rows
    """
    #Predict and prob
    pred = model.predict(data)
    proba = model.predict_proba(data)

    labels = {0: "Edible", 1: "Poisonous"}

    res = []

    for p, pr in zip(pred, proba):
        prob = round(pr[p] * 100, 2)
        res.append({
            "labels": labels[p],
            "prob": float(prob)
        })

    return res
