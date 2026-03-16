from ..registry import save_model, load_model, mlflow_transition_model, mlflow_run
from ..data import get_data
from ..preprocess import tts, preprocess_features
from ..model import define_model

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

import mlflow

@mlflow_run
def train_mlf():
    df = get_data('data')
    X_train, _, y_train, _ = tts(df)
    model = define_model()
    pipe_model = make_pipeline(preprocess_features(), model)

    # train model
    pipe_model.fit(X_train, y_train)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=pipe_model, path='models/tabular/XGBoost')

    # The latest model should be moved to staging
    mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")
    return pipe_model

#python -c 'from XGBoost.interface.main import train_mlf; train_mlf()'
