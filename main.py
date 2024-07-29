#!/usr/bin/env python

from ds_pipe.data_feed import main_feed
from ds_pipe.data_preprocessing import data_preprocessing

# TODO: configurar utils.config - añadir reportes


if __name__ == "__main__":

    file_name = "challenge_edMachina"
    main_feed(f"{file_name}.csv")
    # TODO: descargar un reporte de los datos
    data_preprocessing(f"{file_name}.parquet")
