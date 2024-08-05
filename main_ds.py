#!/usr/bin/env python
import multiprocessing as mp
import time
import warnings

import pendulum

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import data_preprocessing
from ds_pipe.data_preprocessing_mp import data_preprocessing_mp
from ds_pipe.data_research import main_research
from ds_pipe.data_results import main_results

warnings.filterwarnings("ignore")


def main_ds():
    start = time.time()
    file_name = "challenge_edMachina"

    # STEP 1
    main_feed(f"{file_name}.csv")

    # STEP 2
    try:
        data_preprocessing_mp(f"{file_name}.parquet")
    except Exception as e:
        print(
            "Multiprocessing execution failed - Falló la ejecución en procesos paralelos."
        )
        print("Error:", e, end="\n\n")
        data_preprocessing(f"{file_name}.parquet")

    # STEP 3: Not Implemented
    main_research(file_name, download_reports=False)

    # STEP 4
    main_results(f"{file_name}_cleaned.parquet", download_reports=False)

    # Execution Time
    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )
    end = time.time()
    print("Current Execution Time:\n", current_time, end="\n\n")
    print("Total Execution Time:\n", round(end - start, 2), "seconds", end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")


if __name__ == "__main__":
    mp.freeze_support()
    main_ds()
