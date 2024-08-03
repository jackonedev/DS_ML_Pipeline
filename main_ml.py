#!/usr/bin/env python
import time
import warnings
import os

import pendulum
import fire

from ml_pipe.ml_model_1_feature_selector import main_feature_selector
from ml_pipe.ml_model_2_random_forest import main_random_forest
from ml_pipe.ml_model_3_cross_validation import main_cross_validation

from utils.config import ARTIFACTS_PATH

warnings.filterwarnings("ignore")


def main_ml(model_1 = False, model_2 = False, model_3 = False):
    """
    Executes the main machine learning pipeline.
    Args:
        model_1 (bool): Flag indicating whether to perform feature selection using Recursive Feature Elimination (RFE) and to obtain Linear Regression model scores.
        model_2 (bool): Flag indicating whether to execute a simple RandomForestRegressor.
        model_3 (bool): Flag indicating whether to execute a RandomForestRegressor with cross-validation.
    """
    start = time.time()
    dataset_name = "challenge_edMachina"
    feature_name = "grouped_features"
    
    # Model 1 execution is mandatory if the artifacts folder is empty
    if not os.listdir(ARTIFACTS_PATH):
        model_1 = True
    
    if not model_1 and not model_2 and not model_3:
        print("Please, select at least one model to execute.")
        return
    
    if model_1:
        main_feature_selector(dataset_name, feature_name)
    if model_2:
        main_random_forest(dataset_name, feature_name)
    if model_3:
        main_cross_validation(dataset_name, feature_name)

    # Execution Time
    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )
    end = time.time()
    print("Current Execution Time:\n", current_time, end="\n\n")
    print("Total Execution Time:\n", round(end - start, 2), "seconds", end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")


if __name__ == "__main__":
    fire.Fire(main_ml)
