import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import FEATURES_PATH


def load_data(dataset_name, feature_name):
    if feature_name is None:
        raise ValueError("No file name provided")
    if not feature_name.endswith(".parquet"):
        feature_name += ".parquet"
    df = pd.read_parquet(os.path.join(FEATURES_PATH, dataset_name, feature_name))
    return df


def preprocess_data(df, test_size, random_state, scaler):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
