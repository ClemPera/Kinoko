import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from torch import threshold

def define_model():
    """
    Define the model to perform on tabular data
    Model found thanks to AutoML (Flaml), better than with RandomizedSearch on XGBoost
    """
    model = RandomForestClassifier(
        n_estimators=125,
        max_features=0.28106927285321465,
        max_leaf_nodes=3448,  # ⚠️ max_leaves → max_leaf_nodes en sklearn
        criterion='gini',
        n_jobs=-1,
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
    proba = model.predict_proba(data_test)
    pred = (proba[:, 1] >= threshold).astype(int)

    labels = {0: "Edible", 1: "Poisonous"}

    res = []

    for p, pr in zip(pred, proba):
        prob = round(pr[p] * 100, 2)
        res.append(f"{labels[p]} avec {prob:.1f}% proba")

    return res
