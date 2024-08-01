#!/usr/bin/env python
import time

import pendulum

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import data_preprocessing
from ds_pipe.data_research import main_research
from ds_pipe.data_results import main_results


def main():
    start = time.time()
    file_name = "challenge_edMachina"
    main_feed(f"{file_name}.csv")
    # TODO: descargar un reporte de los datos
    data_preprocessing(f"{file_name}.parquet")
    main_research()
    main_results(f"{file_name}_cleaned.parquet")
    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )
    end = time.time()
    print("Current Execution Time:\n", current_time, end="\n\n")
    print("Total Execution Time:\n", round(end - start, 2), "seconds", end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")


if __name__ == "__main__":
    main()
