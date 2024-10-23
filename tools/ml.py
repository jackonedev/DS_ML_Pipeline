import ast
import os
from typing import List

import mlflow
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


def train_model(X_train, y_train, model, **kwargs):
    model = model(**kwargs)
    model.fit(X_train, y_train)
    return model


def obtain_model_1_features(experiment_id: int) -> List[str]:
    """Obtiene las características seleccionadas por el modelo_1

    Args:
        experiment_id (int): Id correspondiente a la interface de mlflow

    Returns:
        List[str]: Lista de las mejores features obtenidas por el modelo_1
    """
    
    experiments = mlflow.search_runs(experiment_ids=[str(experiment_id)], search_all_experiments=True)
    score_cols = ["metrics.train", "metrics.test"]
    cols_name = []
    for col in experiments.columns:
        if col.startswith(score_cols[0]) or col.startswith(score_cols[1]):
            cols_name.append(col)
    run_max_score = experiments[["run_id"] + cols_name].max(axis=1, numeric_only=True)
    id_max_score = run_max_score.idxmax()
    # run_scores = experiments.iloc[id_max_score][cols_name]
    feature_selected = ast.literal_eval(
        experiments.iloc[id_max_score][["params.features_selected"]].values[0]
    )
    feature_selected_ = [f"metrics.coef_{ft}" for ft in feature_selected]
    # se limpian las columnas con coeficiente de regularización Lasso igual a cero
    coef_cero_mask = (experiments.iloc[id_max_score][feature_selected_] == 0).values
    final_features = pd.Series(feature_selected)[~coef_cero_mask].values.tolist()
    return final_features
