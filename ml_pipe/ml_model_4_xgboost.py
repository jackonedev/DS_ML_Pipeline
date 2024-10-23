import os
import time

import mlflow
import xgboost as xgb
from mlflow import log_metric, log_param
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, preprocess_data
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
NOW = str(int(time.time()))
MODEL = xgb.XGBRegressor
early_stop = xgb.callback.EarlyStopping(rounds=10, save_best=True)
HYPERPARAMETERS = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 3,
    "callbacks": [early_stop],
}
TRACKING_ID = 2


def main_xgboost(dataset_name: str, feature_name: str) -> None:
    print("  ML model_4: XGBoost Started!  ".center(88, "."), end="\n\n")

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'{TRACKING_ID}_XGBoost: "{feature_name}"'
    model_registry_name = f"xgboost-{MODEL.__name__}-model".lower()
    # pylint: disable=unused-variable
    model_artifact_path = os.path.join(ARTIFACTS_PATH, model_registry_name, NOW)
    mlflow.set_experiment(experiment_name)

    # .2. Cargar feature-dataset y feature-variables
    df = load_data(dataset_name, feature_name)
    # features = obtain_model_1_features()

    # .3. Construir feature-entrenamiento
    # df = df[features + ["target"]]
    X_train, X_test, y_train, y_test = preprocess_data(
        df, TEST_SIZE, RANDOM_STATE, SCALER
    )

    # .4. Entrenar el modelo
    with mlflow.start_run():
        log_param("model_type", MODEL.__name__)
        log_param("scaler", SCALER.__name__)

        # Crear el modelo XGBoost Regressor
        model = MODEL(**HYPERPARAMETERS)

        # Entrenar el modelo
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Realizar predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluar el rendimiento del modelo
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        print(f"Mean squared error train: {mse_train}")
        print(f"R2 Score train: {r2_train}")
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        print(f"Mean squared error test: {mse_test}")
        print(f"R2 Score test: {r2_test}")

        # .5. Registrar las características y resultados
        log_param("hyperparameters", HYPERPARAMETERS)
        log_metric("train_r2_score", r2_train)
        log_metric("test_r2_score", r2_test)

    print("  ML model_4: XGBoost Done!  ".center(88, "."), end="\n\n")
