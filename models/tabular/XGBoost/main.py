from .data import *
from .model import *
from .preprocess import *
from .registry import *


def main():
    data = get_data("data")
    X_train, _, y_train, _ = tts(data)
    pipeline = preprocess_features()
    model = train_model(X_train, y_train, pipeline)
    save_model(model, "models/tabular/XGBoost")


if __name__ == "__main__":
    main()
