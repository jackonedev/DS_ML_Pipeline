import os
import time

import mlflow
from mlflow import log_metric, log_param
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, obtain_model_1_features, preprocess_data, train_model
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
CV = 5
NOW = str(int(time.time()))
MODEL = ElasticNet
HYPERPARAMETERS = {"alpha": 0.1}
TRACKING_ID = 2

def main_cross_validation(
    dataset_name: str, feature_name: str, select_best_features: bool = True
) -> None:
    print("  ML model_3: Cross Validation Started!  ".center(88, "."), end="\n\n")

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'{TRACKING_ID}_Cross-Validation: "{feature_name}"'
    model_registry_name = f"cross_val-{MODEL.__name__}-model".lower()
    # pylint: disable=unused-variable
    model_artifact_path = os.path.join(ARTIFACTS_PATH, model_registry_name, NOW)
    mlflow.set_experiment(experiment_name)

    # .2. Cargar feature-dataset y feature-variables
    df = load_data(dataset_name, feature_name)
    if select_best_features:
        try:
            features = obtain_model_1_features()
            features += ["target"]
        except Exception as e:
            print("Error:", e, end="\n\n")
            features = df.columns.tolist()
    else:
        features = df.columns.tolist()

    # .3. Construir feature-entrenamiento
    df = df[features]
    X_train, X_test, y_train, y_test = preprocess_data(
        df, TEST_SIZE, RANDOM_STATE, SCALER
    )

    # .4. Entrenar el modelo
    with mlflow.start_run():
        log_param("model_type", MODEL.__name__)
        model = train_model(X_train, y_train, MODEL, **HYPERPARAMETERS)

        # .5. Realizar validación cruzada
        cv = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=make_scorer(r2_score)
        )

        # .6. Registrar métricas y parámetros
        log_metric("mean_cv_r2_score", cv_scores.mean())
        log_metric("std_cv_r2_score", cv_scores.std())
        log_metric("test_r2_score", model.score(X_test, y_test))
        log_metric("train_r2_score", model.score(X_train, y_train))
        log_param("features_selected", features)
        log_param("hyperparameters", HYPERPARAMETERS)
        log_param("random_state", RANDOM_STATE)

        # .7. Registrar el modelo
        mlflow.sklearn.log_model(model, f"{model_registry_name}/{NOW}")

    print("  ML model_3: Cross Validation Done!  ".center(88, "."), end="\n\n")
