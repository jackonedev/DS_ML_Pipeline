import ast
import os
import time
from typing import List

import mlflow
import pandas as pd
from mlflow import log_metric, log_param
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, preprocess_data
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
CV = 5
NOW = str(int(time.time()))


def feature_selection() -> List[str]:
    experiments = mlflow.search_runs(experiment_ids=["1"], search_all_experiments=True)
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


def train_model(X_train, y_train, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def main_random_forest(dataset_name: str, feature_name: str) -> None:
    print("main_random_forest")

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'Random Forest: "{feature_name}"'
    model_registry_name = "sk-learn-random-forest-reg-model"
    model_artifact_path = os.path.join(ARTIFACTS_PATH, model_registry_name, NOW)
    if not os.path.exists(model_artifact_path):
        os.makedirs(model_artifact_path)
    mlflow.set_experiment(experiment_name)

    # .2. Cargar feature-dataset y feature-variables
    df = load_data(dataset_name, feature_name)
    features = feature_selection()

    # .3. Construir feature-entrenamiento
    df = df[features + ["target"]]
    X_train, X_test, y_train, y_test = preprocess_data(
        df, TEST_SIZE, RANDOM_STATE, SCALER
    )

    # .4. Seleccionar parametros
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
    }

    # .5. Entrenar el modelo
    with mlflow.start_run():
        log_param("model_type", "RandomForestClassifier")
        model = train_model(X_train, y_train, **hyperparameters)
        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        # .6. Registrar las características y resultados
        log_param("features", features)
        log_param("hyperparameters", hyperparameters)
        log_metric("train_accuracy", model.score(X_train, y_train))
        log_metric("test_accuracy", accuracy)

        # .7. Registrar el modelo
        mlflow.sklearn.log_model(model, f"{model_registry_name}/{NOW}")
        log_param("features_selected", features)


if __name__ == "__main__":
    dataset_name = "challenge_edMachina"
    feature_name = "grouped_features"
    main_random_forest(dataset_name, feature_name)
    print("done!")
