import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


def tts(df: pd.DataFrame, test_size=0.3, random_state=3):
    """
    Train test split
    """
    # X and y ready
    X = df.drop(columns=['class','gill_spacing','stem_root','stem_surface',
                         'veil_type','veil_color','spore_print_color','scientific_name'])
    y = df['class']

    # TTS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    #X_test_noisy = add_noise_to_dataset(X_test)

    return X_train, X_test, y_train, y_test


def preprocess_features():
    """
    Preprocess pipeline to fit_transform() on train data
    """
    # Preprocess pipeline
    num_transformer = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    cat_transformer = make_pipeline(SimpleImputer(strategy='constant', fill_value='u'),
                                    OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))

    # preprocess all
    preproc_basic = make_column_transformer(
        (num_transformer, make_column_selector(dtype_include=np.number)),
        (cat_transformer, make_column_selector(dtype_exclude=np.number)),
        remainder='drop'
    ).set_output(transform="pandas")

    return preproc_basic
