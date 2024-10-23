#!/usr/bin/env python
import time
import warnings

import fire
import pendulum

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import data_preprocessing
from ds_pipe.data_preprocessing_mp import data_preprocessing_mp
from ds_pipe.data_research import main_research
from ds_pipe.data_results import main_results
from utils.config import DATASET_NAME, COURSE_NAMES

warnings.filterwarnings("ignore")



def main_ds(
    download_reports: bool = False,
    course_name_filter: bool = True
) -> None:
    """
    Main function to execute the data science pipeline.
    Args:
        download_reports (bool): Flag to indicate whether to download reports. Defaults to False.
        course_name_filter (bool): Flag to indicate whether to filter by course name. Defaults to True.
    Returns:
        None
    Example:
        main_ds(
            download_reports=True,
            course_name_filter=True
        )
    """
    
    
    
    start = time.time()

    # STEP 1
    main_feed(f"{DATASET_NAME}.csv")

    # STEP 2
    # pylint: disable=broad-exception-caught
    try:
        data_preprocessing_mp(f"{DATASET_NAME}.parquet")
    except Exception as e:
        print(
            "Multiprocessing execution failed - Falló la ejecución en procesos paralelos."
        )
        print("Error:", e, end="\n\n")
        data_preprocessing(f"{DATASET_NAME}.parquet")

    # STEP 3:
    main_research(DATASET_NAME, download_reports=download_reports)

    # STEP 4
    if course_name_filter:
        course_name_filter = COURSE_NAMES
    else:
        course_name_filter = []
    main_results(
        f"{DATASET_NAME}_cleaned.parquet",
        download_reports=download_reports,
        course_name_filter=course_name_filter,
    )

    # Execution Time
    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )
    end = time.time()
    print("Current Execution Time:\n", current_time, end="\n\n")
    print("Total Execution Time:\n", round(end - start, 2), "seconds", end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")


if __name__ == "__main__":

    fire.Fire(main_ds)
