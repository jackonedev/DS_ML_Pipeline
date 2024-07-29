"""
Segundo step en el pipeline de DS.

Este step se encarga de la limpieza de los datos.
Descarga el archivo parquet con los datos limpios
en el directorio ds_pipe/data/parquet.
"""

from typing import Union

import pandas as pd

from tools.business import eliminar_cursada
from utils.config import PARQUET_PATH


def data_preprocessing(parquet_name: Union[str, None] = None) -> None:
    """
    STEPS:
      - Lee un archivo parquet localizado en el directorio datasets/parquet.
      - Elimina registros duplicados.
      - Elimina registros cuyas fechas sean todas nulas (particion inactiva).
      - Elimina outliers que no tengan 100 puntos posibles.
      - Elimina columnas irrelevantes.
      - Imputa valores faltantes.
      - Guarda una copia en formato parquet con el mismo nombre más "_cleaned".

    Args:
    parquet_name (Union[str, None]): Nombre del archivo parquet a leer ubicado en datasets/parquet.
    """

    print("  Step 2: Data Preprocessing Started  ".center(88, "."), end="\n\n")
    # Handling edge cases
    if parquet_name is None:
        raise ValueError("No file name provided")

    # Read Local Parquet file
    if not parquet_name.endswith(".parquet"):
        parquet_name += ".parquet"
    df = pd.read_parquet(f"{PARQUET_PATH}/{parquet_name}")

    # Eliminar registros duplicados
    df = df.drop_duplicates()

    # Eliminar registro cuyas fechas sean todas nulas (particion inactiva)
    datetime_columns = [
        col
        for col in df.columns
        if df[col].dtype == "datetime64[ns, America/Buenos_Aires]"
    ]
    df = df.dropna(subset=datetime_columns, how="all")

    # Eliminar outliers que no tengan 100 puntos posibles
    df_outlier = df[(df["points_possible"] == 0) | (df["points_possible"] == 10)]
    df = eliminar_cursada(df, df_outlier)

    # Eliminar columnas irrelevantes
    df = df.drop(
        columns=[
            # "course_name",
            # "ass_due_at",
            "s_created_at",
            # "s_graded_at",
            "periodo",
            "points_possible",
        ]
    )
    # pylint: disable=W0511
    # TODO: Imputación
    # .1.

    # Descarga en parquet
    parquet_name = parquet_name.split(".")[0] + "_cleaned.parquet"
    df.to_parquet(f"{PARQUET_PATH}/{parquet_name}")
    print(f"File saved as {parquet_name} in {PARQUET_PATH}", end="\n\n")
    print("  Step 2: Data Preprocessing ##Completed  ".center(88, "."), end="\n\n")
