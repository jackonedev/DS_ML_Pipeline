import os
import time

import mlflow
from mlflow import log_metric, log_param
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, obtain_model_1_features, preprocess_data, train_model
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
NOW = str(int(time.time()))
MODEL = RandomForestRegressor
HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
}


def main_random_forest(dataset_name: str, feature_name: str) -> None:
    print("  ML model_2: Random Forest Started!  ".center(88, "."), end="\n\n")

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'Random Forest: "{feature_name}"'
    model_registry_name = f"sk-learn-{MODEL.__name__}-model".lower()
    # pylint: disable=unused-variable
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
        model = train_model(X_train, y_train, MODEL, **HYPERPARAMETERS)
        train_r2_score = model.score(X_train, y_train)
        test_r2_score = model.score(X_test, y_test)

        # .5. Registrar las características y resultados
        log_param("features", features)
        log_param("hyperparameters", HYPERPARAMETERS)
        log_metric("train_r2_score", train_r2_score)
        log_metric("test_r2_score", test_r2_score)
        print("train_r2_score: ", train_r2_score)
        print("test_r2_score: ", test_r2_score)

        # .6. Registrar el modelo
        mlflow.sklearn.log_model(model, f"{model_registry_name}/{NOW}")

    print("  ML model_2: Random Forest Done!  ".center(88, "."), end="\n\n")
