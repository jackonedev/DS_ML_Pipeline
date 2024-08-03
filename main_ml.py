#!/usr/bin/env python
import time
import warnings

import pendulum

from ml_pipe.ml_model_1_feature_selector import main_feature_selector
from ml_pipe.ml_model_2_random_forest import main_random_forest

warnings.filterwarnings("ignore")


def main_ml():
    start = time.time()
    dataset_name = "challenge_edMachina"
    feature_name = "grouped_features"
    # main_feature_selector(dataset_name, feature_name)
    main_random_forest(dataset_name, feature_name)

    # Execution Time
    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )
    end = time.time()
    print("Current Execution Time:\n", current_time, end="\n\n")
    print("Total Execution Time:\n", round(end - start, 2), "seconds", end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")


if __name__ == "__main__":
    main_ml()
