import ast
import os
import time
from typing import List

import mlflow
import pandas as pd
from mlflow import log_metric, log_param
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, obtain_model_1_features, preprocess_data, train_model
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
CV = 5
NOW = str(int(time.time()))
MODEL = RandomForestClassifier
HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
}


def main_cross_validation(dataset_name: str, feature_name: str) -> None:
    print("  ML model_3: Cross Validation Started!  ".center(88, "."), end="\n\n")

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'Cross-Validation: "{feature_name}"'
    model_registry_name = f"cross_val-{MODEL.__name__}-model".lower()
    model_artifact_path = os.path.join(ARTIFACTS_PATH, model_registry_name, NOW)
    mlflow.set_experiment(experiment_name)

    # .2. Cargar feature-dataset y feature-variables
    df = load_data(dataset_name, feature_name)
    features = obtain_model_1_features()

    # .3. Construir feature-entrenamiento
    df = df[features + ["target"]]
    X_train, X_test, y_train, y_test = preprocess_data(
        df, TEST_SIZE, RANDOM_STATE, SCALER
    )

    # .4. Entrenar el modelo
    with mlflow.start_run():
        log_param("model_type", MODEL.__name__)
        model = train_model(X_train, y_train, MODEL)

        # .5. Realizar validación cruzada
        cv = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=make_scorer(accuracy_score)
        )

        # .6. Registrar métricas y parámetros
        log_metric("mean_cv_accuracy", cv_scores.mean())
        log_metric("std_cv_accuracy", cv_scores.std())
        log_metric("test_accuracy", model.score(X_test, y_test))
        log_metric("train_accuracy", model.score(X_train, y_train))
        log_param("features_selected", features)
        log_param("hyperparameters", HYPERPARAMETERS)

        # .7. Registrar el modelo
        mlflow.sklearn.log_model(model, f"{model_registry_name}/{NOW}")

    print("  ML model_3: Cross Validation Done!  ".center(88, "."), end="\n\n")
