#!/usr/bin/env python

import pendulum

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import data_preprocessing

# TODO: configurar utils.config - añadir reportes


if __name__ == "__main__":

    file_name = "challenge_edMachina"
    main_feed(f"{file_name}.csv")
    # TODO: descargar un reporte de los datos
    data_preprocessing(f"{file_name}.parquet")

    current_time = (
        pendulum.now().set(microsecond=0, second=0).format("dddd, MMMM Do YYYY, h:mm A")
    )

    print("Current Execution Time:\n", current_time, end="\n\n")
    print("  Main Pipeline Completed Successfully  ".center(88, "."), end="\n\n")
