"""
Segundo step en el pipeline de DS.

Este step se encarga de la limpieza de los datos.
Realiza el tratamiento correspondiente para las columnas de fechas,
Descarga el archivo parquet con los datos limpios
en el directorio ds_pipe/data/parquet.
"""

from typing import Union

import pandas as pd

from tools.business import eliminar_cursada
from tools.dates import round_timedelta
from utils.config import PARQUET_PATH


def data_preprocessing(parquet_name: Union[str, None] = None) -> None:
    """
    STEPS:
      - Lee un archivo parquet localizado en el directorio datasets/parquet.
      - Elimina registros duplicados.
      - Elimina registros cuyas fechas sean todas nulas (particion inactiva).
      - Elimina outliers que no tengan 100 puntos posibles.
      - Agregación de columnas de fechas.
      - Agregación de columnas de resumen actividad (not implemented).
      - Elimina columnas irrelevantes.
      - Ordenar columnas.
      - Eliminar duplicados encontrados luego de la imputacion.
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

    # Eliminar outliers que no tengan 100 puntos posibles - 'points_possible'
    df_outlier = df[(df["points_possible"] == 0) | (df["points_possible"] == 10)]
    df = eliminar_cursada(df, df_outlier)

    # Agregacion de las fechas
    # Unificar la columna de fechas en una sola 'activity_date'
    # Llenar los NaN en la columna 'ass_unlock_at' basado en el mismo 'course_uuid'
    # Luego, los elementos restantes por el elemento más cercano.
    # Crear la columna 'activity_unlock_delta', distancia entre unlock y el alumno submittea
    df["activity_date"] = (
        df["s_submitted_at"]
        .combine_first(df["fecha_mesa_epoch"])
        .combine_first(df["ass_due_at"])
    )
    df["ass_unlock_at"] = (
        df.groupby("course_uuid")["ass_unlock_at"]
        .transform(lambda x: x.ffill().bfill())
        .ffill()
        .bfill()
    )
    df["activity_unlock_delta"] = (df["activity_date"] - df["ass_unlock_at"]).apply(
        round_timedelta
    )

    # Agregacion activity_overall
    # NotImplemented ... yet

    # Eliminar columnas irrelevantes
    df = df.drop(
        columns=[
            "periodo",
            "fecha_mesa_epoch",
            "ass_due_at",
            # "ass_unlock_at",
            "s_submitted_at",
            "ass_created_at",
            "ass_lock_at",
            "s_created_at",
            "s_graded_at",
            "points_possible",
            "legajo",
            "sub_uuid",
            "assignment_id",
            # "nombre_examen",
            # "ass_name",
            # "ass_name_sub",
        ]
    )

    # Ordenar columnas
    # Definicion del orden deseado de las columnas
    # Reindexar el DataFrame con el nuevo orden de columnas
    order = [
        col
        for col in df.columns
        if col not in ["ass_unlock_at", "activity_date", "activity_unlock_delta"]
    ] + ["ass_unlock_at", "activity_date", "activity_unlock_delta"]
    df = df.reindex(columns=order)

    # Eliminar filtrados duplicados luego de la imputacion
    df = df.drop_duplicates()

    # Descarga en parquet
    parquet_name = parquet_name.split(".")[0] + "_cleaned.parquet"
    df.to_parquet(f"{PARQUET_PATH}/{parquet_name}")
    print(f"File saved as {parquet_name} in {PARQUET_PATH}", end="\n\n")
    print("  Step 2: Data Preprocessing ##Completed  ".center(88, "."), end="\n\n")
