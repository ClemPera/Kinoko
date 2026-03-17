import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

def define_model() -> RandomForestClassifier:
    """
    Define the model to perform on tabular data

    Returns:
        Classifier model
    """
    model = RandomForestClassifier(
        n_estimators=125,
        max_features=0.28106927285321465,
        max_leaf_nodes=3448,  # ⚠️ max_leaves → max_leaf_nodes en sklearn
        criterion='gini',
        n_jobs=-1,
        random_state=3)

    return model


def train_model(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                pipeline: BaseEstimator) -> Pipeline:
    """
    Training the model

    Args:
        - X_train, y_train : from tts()
        - pipeline: from preprocess_features()

    Returns: Fitted pipeline
    """
    # call the function made before
    model = define_model()

    # Pipeline with data preprocessed and model
    pipe_model = make_pipeline(pipeline, model)

    # train model
    pipe_model.fit(X_train, y_train)

    return pipe_model


def predict(model: Pipeline, data: pd.DataFrame) -> tuple[bool, float]:
    """
    Prediction function, gives out the predicted class and the associated prob
    Args:
        - model : model trained before
        - data : can be a single data to predict

    Returns:
        tuple of predicted class and probability
    """
    # Predict and prob

    proba = model.predict_proba(data)
    pred = (proba[:, 1] >= 0.35).astype(int)

    return bool(pred), float(proba[0][pred])


def predict_multiple(model: Pipeline, data: pd.DataFrame):
    """
    Prediction function, gives out the predicted class and the associated prob
    Args:
        - model : model trained before
        - data : can be a single data to predict

    Returns:
        List of predicted labels and probabilities
    """
    #Predict and prob
    proba = model.predict_proba(data)
    pred = (proba[:, 1] >= 0.35).astype(int)

    labels = {0: "Edible", 1: "Poisonous"}

    res = []

    for p, pr in zip(pred, proba):
        prob = round(pr[p] * 100, 2)
        res.append({
            "labels": labels[p],
            "prob": float(prob)
        })

    return res
