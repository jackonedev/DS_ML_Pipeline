"""
Primer Step en el pipeline de DS

Este modulo se encarga de la lectura de los datos, ya sea de un archivo o de una base de datos.
Hace un relevamiento en la consistencia de los tipos de datos, principalmente la existencia de fechas.
Actualiza los tipos de datos si es necesario.
Descarga una copia en formato parquet.
"""

import os

import pandas as pd

from utils.config import DATASETS_PATH, PARQUET_PATH


def main_feed():
    # File name
    file_name = os.path.join(DATASETS_PATH, "challenge_edMachina.csv")
    # Read data
    df = pd.read_csv(file_name, sep=';')
    # Check data types
    print(df.dtypes)
    # Check for date columns
    # print(df.select_dtypes(include=["datetime64"]).columns)
    # Save a copy in parquet format
    df.to_parquet(f"{PARQUET_PATH}/covid_data.parquet")