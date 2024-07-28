"""
Segundo step en el pipeline de DS.

Este step se encarga de la limpieza de los datos.
Actualiza el archivo parquet con los datos limpios.
"""

import pandas as pd

from utils.config import PARQUET_PATH

fn = "challenge_edMachina.parquet"

df = pd.read_parquet(f"{PARQUET_PATH}/{fn}")


def hello():
    print("Hello from data_preprocessing.py\n")
    print(df.dtypes)
    print(df.shape)
