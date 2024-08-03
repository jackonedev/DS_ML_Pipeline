import json
import os
import time

import mlflow
import pandas as pd
from mlflow import log_artifacts, log_metric, log_param
from scipy.signal import argrelextrema
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tools.ml import load_data, preprocess_data
from tools.serializers import custom_serializer
from utils.config import ARTIFACTS_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.15
SCALER = StandardScaler
SCORE_FUNC = mutual_info_regression
CV = 5
NOW = str(int(time.time()))


def feature_selection(X_train, y_train, k=10):
    selector = SelectKBest(score_func=SCORE_FUNC, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected, selector


def train_model(X_train, y_train):
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    return model


def main_feature_selector(dataset_name: str, feature_name: str) -> None:

    # .1. Inicia un nuevo experimento
    feature_name = f"{dataset_name}_{feature_name}"
    experiment_name = f'Feature Selection: "{feature_name}"'
    model_registry_name = "sk-learn-lasso-feature-selector-model"
    model_artifact_path = os.path.join(ARTIFACTS_PATH, model_registry_name, NOW)
    if not os.path.exists(model_artifact_path):
        os.makedirs(model_artifact_path)
    mlflow.set_experiment(experiment_name)

    # .2. Cargar feature y procesarla
    df = load_data(dataset_name, feature_name)
    X_train, X_test, y_train, y_test = preprocess_data(
        df, TEST_SIZE, RANDOM_STATE, SCALER
    )
    df = df.iloc[:, :-1]

    # .3. Recursive Feature Elimination with correlated features
    clf = LogisticRegression()
    cv = StratifiedKFold(CV)
    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)
    cv_results = pd.DataFrame(rfecv.cv_results_)

    # .4. Obtención de los máximos locales del mean accuracy de la validación cruzada
    # mirar tools.ml_plots
    maxima_indices = argrelextrema(
        cv_results["mean_test_score"].values, comparator=lambda a, b: a > b
    )[0]
    maxima_values = cv_results["mean_test_score"].iloc[maxima_indices]
    maxima_values.index = maxima_values.index + 1

    # .5. Ejecucion del MLflow Tracking
    # Itera sobre los máximos locales de cantidad de features
    # Registrar los resultados en MLflow
    # Entrenamiento de modelo Lasso para calcular coeficientes de regularización
    # Almacenar localmente y registrar los resultados en MLflow
    # Entrenamiento de modelo Linear Regression y ElasticNet
    # Registrar los resultados en MLflow
    ks = list(maxima_values.index)
    for k in ks:
        with mlflow.start_run(run_name=f"{NOW}/k_{k}") as run:
            # 5.1
            run_id = run.info.run_id
            log_param("k", k)
            log_param("scaler", SCALER.__name__)
            log_param("random state", RANDOM_STATE)
            log_param("cv folds", CV)
            print(f"Ejecutando Feature Selector con {k} features")
            X_train_selected, selector = feature_selection(X_train, y_train, k=k)
            features_selected = list(df.columns[selector.get_support()])
            features_scores = selector.scores_[selector.get_support()]
            score_func_name = selector.score_func.__name__
            log_param("features_selected", features_selected)
            log_param("score_func_name", score_func_name)
            [
                log_metric(f"score_{ft}", ft_score)
                for ft, ft_score in zip(features_selected, features_scores)
            ]

            # 5.2 Entrenar el modelo Lasso y registrar los resultados
            X_test_selected = selector.transform(X_test)
            model = train_model(X_train_selected, y_train)
            train_score_lasso = model.score(X_train_selected, y_train)
            test_score_lasso = model.score(X_test_selected, y_test)
            features_coef = model.coef_
            log_metric("train_score_lasso", train_score_lasso)
            log_metric("test_score_lasso", test_score_lasso)
            [
                log_metric(f"coef_{ft}", ft_coef)
                for ft, ft_coef in zip(features_selected, features_coef)
            ]

            results = {
                "run_id": run_id,
                "features_selected": features_selected,
                "features_scores": features_scores,
                "score_func_name": score_func_name,
                "train_score": train_score_lasso,
                "test_score": test_score_lasso,
                "features_coef": features_coef,
            }

            with open(os.path.join(model_artifact_path, f"k_{k}.json"), "w") as f:
                json.dump(results, f, default=custom_serializer)

            # 5.3 Train a Linear Model and ElasticNet
            lin_reg = LinearRegression()
            lin_reg.fit(X_train_selected, y_train)
            train_score_lin_reg = lin_reg.score(X_train_selected, y_train)
            test_score_lin_reg = lin_reg.score(X_test_selected, y_test)
            log_metric("train_score_lin_reg", train_score_lin_reg)
            log_metric("test_score_lin_reg", test_score_lin_reg)
            elastic_net = ElasticNet(alpha=0.1)
            elastic_net.fit(X_train_selected, y_train)
            train_score_elastic_net = elastic_net.score(X_train_selected, y_train)
            test_score_elastic_net = elastic_net.score(X_test_selected, y_test)
            log_metric("train_score_elastic_net", train_score_elastic_net)
            log_metric("test_score_elastic_net", test_score_elastic_net)
            # mlflow.sklearn.log_model(model, "model")

    log_artifacts(model_artifact_path)
    print(f"Done!")
