"""
Segundo step en el pipeline de DS.

Este step se encarga de la limpieza de los datos.
Realiza el tratamiento correspondiente para las columnas de fechas,
y realiza las agregaciones correspondientes para la creación de nuevas columnas.
Estandariza las particiones para que todas representen una sola actividad.
Ejecuta el proceso de labelización de columnas categóricas con multiples valores similares.
Realiza el formateo final del dateset y guarda el dataset limpio en formato parquet en la carpeta datasets/parquet.
Guarda el diccionario de leyenda en formato JSON en la carpeta datasets/parquet.
"""

import json
import re
from typing import Union

import numpy as np
import pandas as pd

from tools.business import eliminar_cursada
from tools.dates import round_timedelta
from utils.config import PARQUET_PATH


def data_preprocessing(parquet_name: Union[str, None] = None) -> None:
    """
    DATA PREPROCESSING STEP
    EXECUTION PROCESSES:
    - .1. Handling edge cases
    - .2. Read Local Parquet file
    - .3. Eliminar registro cuyas fechas sean todas nulas (particion inactiva)
    - .4. Eliminar outliers que no tengan 100 puntos posibles - 'points_possible'
    - .5. Eliminar registros redundantes cuando nota_parcial y score sean NaN
    - .6. Eliminar course_name con inconsistencias en las notas
    - .7. Agregacion de las fechas
    - .8. Eliminar columnas irrelevantes
    - .9. Dividir particiones que tengan 'fecha_mesa_epoch' y 's_submitted_at' compartido
    - .10. Labelización
    - .11. Agregación de columna labelizada 'ass_name_label'
    - .12. Tratamiento de las columnas de notas y scores
    - .13. Ajustes Finales
    - .14. Descarga de datos

    Args:
    parquet_name (Union[str, None]): Nombre del archivo parquet a leer ubicado en datasets/parquet.
    """

    print("  Step 2: Data Preprocessing Started  ".center(88, "."), end="\n\n")
    # .1. Handling edge cases
    if parquet_name is None:
        raise ValueError("No file name provided")
    if not parquet_name.endswith(".parquet"):
        parquet_name += ".parquet"

    # .2. Read Local Parquet file
    # Leer fichero y eliminar registros duplicados
    df = pd.read_parquet(f"{PARQUET_PATH}/{parquet_name}").drop_duplicates()

    # .3. Eliminar registro cuyas fechas sean todas nulas (particion inactiva)
    datetime_columns = [
        col
        for col in df.columns
        if df[col].dtype == "datetime64[ns, America/Buenos_Aires]"
    ]
    df = df.dropna(subset=datetime_columns, how="all")

    # .4. Eliminar outliers que no tengan 100 puntos posibles - 'points_possible'
    df_outlier = df[(df["points_possible"] == 0) | (df["points_possible"] == 10)]
    df = eliminar_cursada(df, df_outlier)

    # .5. Eliminar registros redundantes cuando nota_parcial y score sean NaN
    # Eliminar registros redundantes cuando nota_parcial y score sean NaN
    # Borrar datos sobrantes de las otras columnas
    df = df.drop(df.loc[df.score.isna() & df.nota_parcial.isna()].index)
    df.loc[
        df[df.ass_name_sub.notna() & df.score.isna()].index,
        ["ass_name_sub", "submission_type"],
    ] = np.nan

    # .6. Eliminar course_name con inconsistencias en las notas
    # "Upgradable cohesive circuit" y "Visionary exuding knowledge user"
    # nota promedio de "nota_parcial" y de "score" nulo, y
    # promedio de "nota_final_materia" igual a 10.
    # No hay datos para sustentar la nota final.
    course_grouped = (
        df.groupby("course_name")
        .apply(
            lambda x: pd.Series(
                {
                    "mean_nota_parcial": x["nota_parcial"].mean(),
                    "mean_score": x["score"].mean(),
                    "nota_final_materia": x["nota_final_materia"].mean(),
                }
            )
        )
        .reset_index()
    )
    outlier_course_name = course_grouped[
        course_grouped.mean_score.isna() & course_grouped.mean_nota_parcial.isna()
    ].course_name
    df = df.drop(df[df.course_name.isin(outlier_course_name)].index)

    # .7. Agregacion de las fechas
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

    # .8. Eliminar columnas irrelevantes
    df = df.drop(
        columns=[
            "periodo",
            "ass_created_at",
            "fecha_mesa_epoch",
            "ass_due_at",
            "s_submitted_at",
            "ass_lock_at",
            "s_created_at",
            "s_graded_at",
            "points_possible",
            "legajo",
            "sub_uuid",
            "assignment_id",
        ]
    ).drop_duplicates()

    # .9. Dividir particiones que tengan 'fecha_mesa_epoch' y 's_submitted_at' compartido
    # Se duplican los registros que contengan 'nombre_examen' y 'ass_name_sub'
    # Se crea una copia, y se completan con NaN los registros relacionados a parciales
    # Se crea otra copia, se eliminan las columnas relacionadas a ass_sub, y se eliminan duplicados
    # Se concatenan los resultados
    # Encontrar registros que comparten entrada en 'nombre_examen' y 'ass_name_sub'
    dual_reg_df = df[df.nombre_examen.notna() & df.ass_name_sub.notna()]
    ass_name_sub_corrected = dual_reg_df.copy()
    ass_name_sub_corrected[["nombre_examen", "nota_parcial"]] = np.nan
    nombre_examen_corrected = dual_reg_df.copy()
    nombre_examen_corrected = dual_reg_df.drop(
        columns=["ass_name_sub", "score", "submission_type", "activity_date"]
    ).drop_duplicates()
    new_shape_ = df.shape[0] + nombre_examen_corrected.shape[0]
    nombre_examen_corrected.loc[
        :, ["ass_name_sub", "score", "submission_type", "activity_date"]
    ] = dual_reg_df[["activity_date", "submission_type"]]
    df = pd.concat(
        [df.drop(dual_reg_df.index), ass_name_sub_corrected, nombre_examen_corrected]
    )
    assert df.shape[0] == new_shape_, "Not the expected number of rows"

    # .10. Labelización
    # Crear diccionario de leyenda
    # 'submited_type' update 'nombre_examen' en 'submission_type'
    # Quitar los 'ass_name' que se infiltran en 'nombre_examen'
    # 'sub_ass_name' update
    # Unir 'ass_name' con 'ass_name_sub'
    # Unificar registros por patrones regex
    # Actualizar diccionario de leyenda
    legend = {"nombre_examen": set(df.nombre_examen.dropna().tolist())}
    df.loc[df.nombre_examen.notna(), "submission_type"] = "nombre_examen"
    df.loc[
        df.nombre_examen.notna() & df.ass_name.notna() & df.ass_name_sub.isna(),
        "ass_name",
    ] = np.nan

    ass_name_type = df["ass_name"].combine_first(df["ass_name_sub"]).dropna()
    patterns = {
        r"(?i).+\s*\[api1\]\s*": "[API1]",
        r"(?i).+\s*\[ap1\]\s*": "[API1]",
        r"(?i).+\s*\[api2\]\s*": "[API2]",
        r"(?i).+\s*\[api3\]\s*": "[API3]",
        r"(?i).+\s*\[api4\]\s*": "[API4]",
        r"(?i).+\s*\[tp1\]\s*": "[TP1]",
        r"(?i).+\s*\[tp2\]\s*": "[TP2]",
        r"(?i).+\s*\[tp3\]\s*": "[TP3]",
        r"(?i).+\s*\[tp4\]\s*": "[TP4]",
        r"(?i)\s*\[tp4\]\s*.+": "[TP4]",
        r"(?i).+\s*\[ed3\]\s*": "[ED3]",
        r"(?i).+\s*\[ed4\]\s*": "[ED4]",
        r"(?i).+\s*-\s*m1\s*": "[M1]",
        r"(?i).+\s*-\s*m2\s*": "[M2]",
        r"(?i).+\s*-\s*m3\s*": "[M3]",
        r"(?i).+\s*-\s*m4\s*": "[M4]",
        r"(?i).+\[ta1\]\s*": "[TA1]",
        r"(?i).+\[ta2\]\s*": "[TA2]",
        r"(?i).+\[ta3\]\s*": "[TA3]",
    }

    # TODO: OPTIMIZAR EJECUCION
    # Buscar coincidencias y almacenarlas en el diccionario
    # Funcion interna: HILO 1
    def update_legend(patterns, ass_name_type, legend):
        for pattern in patterns:
            regex = re.compile(pattern)
            legend[patterns[pattern]] = set(
                ass_name_type[ass_name_type.str.contains(regex)].tolist()
            )

    # Funcion interna: HILO 2

    def replace_patterns(patterns, ass_name_type, legend):
        return ass_name_type.replace(patterns, regex=True)

    # TODO: OPTIMIZAR EJECUCION
    # PROCESO 1
    # Buscar coincidencias y almacenarlas en el diccionario
    for pattern in patterns:
        regex = re.compile(pattern)
        legend[patterns[pattern]] = set(
            ass_name_type[ass_name_type.str.contains(regex)].tolist()
        )
    # PROCESO 2
    ass_name_label = ass_name_type.replace(patterns, regex=True)

    # CONTINUE
    # pylint: disable=unused-variable
    ass_name_other_mask = ass_name_label.isin(
        (
            other_ass_name := ass_name_label.value_counts()[
                (
                    ass_name_threshold := (
                        ass_name_label.value_counts() < 15
                    ).values.tolist()
                )
            ].index
        )
    )
    ass_name_label[ass_name_other_mask] = "[OTHER]"
    legend.update({"[OTHER]": set(other_ass_name)})
    ass_name_rest_value_counts_mask = ~pd.Series(
        ass_name_label.value_counts().index
    ).isin(legend.keys())
    ass_name_rest_mask = ass_name_label.isin(
        ass_name_label.value_counts()[ass_name_rest_value_counts_mask.values].index
    )
    ass_name_label[ass_name_rest_mask] = ass_name_label[ass_name_rest_mask].apply(
        lambda x: f"[{x}]"
    )
    legend.update(
        dict(
            zip(
                [
                    cat
                    for cat in ass_name_label.value_counts()[
                        ass_name_rest_value_counts_mask.values
                    ].index
                ],
                [
                    set([cat])
                    for cat in ass_name_label.value_counts()[
                        ass_name_rest_value_counts_mask.values
                    ].index
                ],
            )
        )
    )

    # .11. Agregación de columna labelizada 'ass_name_label'
    assert (
        df["nombre_examen"].isna().sum() == ass_name_label.shape[0]
    ), "Not correctly duplicated the rows in step 8"
    assert (
        df["nombre_examen"].notna().sum() + ass_name_label.shape[0] == df.shape[0]
    ), "Not the expected number of rows"
    assert (
        df["nombre_examen"].combine_first(ass_name_label).shape[0] == df.shape[0]
    ), "Not the expected number of rows"
    assert (
        ass_name_label.shape[0] + df[df.nombre_examen.notna()].shape[0] == df.shape[0]
    ), "Not the expected number of rows"
    df["ass_name_label"] = df["nombre_examen"].combine_first(ass_name_label).values

    # .12. Tratamiento de las columnas de notas y scores
    # Ajuste de escala de notas de 0 a 100 en 'nota_parcial' y 'nota_final_materia'
    # Unificar las columnas 'score' y 'nota_parcial' en 'score'
    df["nota_parcial"], df["nota_final_materia"] = (
        df["nota_parcial"] * 10,
        df["nota_final_materia"] * 10,
    )
    df["score"] = df["score"].fillna(df["nota_parcial"])

    # .13. Ajustes Finales
    # Drop the last columns. Eliminar duplicados infiltrados luego de la imputacion
    # Definicion del orden deseado de las columnas
    # Reindexar el DataFrame con el nuevo orden de columnas
    # Formatear el indice del dataset
    df = df.drop(
        columns=[
            "nota_parcial",
            # "course_name",
            "ass_name",
            "nombre_examen",
            "ass_name_sub",
        ]
    ).drop_duplicates()

    selected_init_cols = ["user_uuid", "course_uuid", "particion"]
    selected_final_cols = [
        "ass_name_label",
        "submission_type",
        "ass_unlock_at",
        "activity_date",
        "activity_unlock_delta",
        "score",
        "nota_final_materia",
    ]
    order = (
        selected_init_cols
        + [
            col
            for col in df.columns
            if col not in selected_init_cols + selected_final_cols
        ]
        + selected_final_cols
    )
    df = df.reindex(columns=order)
    df = df.sort_index().reset_index(drop=True)
    assert df.duplicated().shape[0] != 0, "Duplicated elements existance"

    # .14. Descarga de datos
    # Descarga en parquet del dataset en datasets/parquet
    # Descarga en JSON de legend en  datasets/parquet
    parquet_name = parquet_name.split(".")[0] + "_cleaned.parquet"
    json_name = parquet_name.split(".")[0] + "_legend.json"
    df.to_parquet(f"{PARQUET_PATH}/{parquet_name}")
    print(f"File saved as {parquet_name} in {PARQUET_PATH}", end="\n\n")
    with open(f"{PARQUET_PATH}/{json_name}", "w", encoding="utf-8") as fp:
        json.dump({k: list(v) for k, v in legend.items()}, fp)

    print("  Step 2: Data Preprocessing Completed  ".center(88, "."), end="\n\n")
