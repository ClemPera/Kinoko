import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

from XGBoost.data import get_data, get_data_reduced
from XGBoost.preprocess import preprocess_features, tts
from utils import add_noise_to_dataset

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

    # TODO: add something to save model?

    return pipe_model


def predict(model, data_test):
    """
    Prediction function, gives out the predicted class and the associated prob
    - model : model trained before
    - data_test : can be a single data to test, or a dataframe with xx rows
    """
    #Predict and prob
    pred = model.predict(data_test)
    proba = model.predict_proba(data_test)

    labels = {0: "Edible", 1: "Poisonous"}

    res = []

    for p, pr in zip(pred, proba):
        prob = pr[p] * 100
        res.append(f"{labels[p]} avec {prob:.1f}% proba")

    return res
