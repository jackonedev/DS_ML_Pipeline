"""
Tercer step en el pipeline de DS.

Este step se encarga de la exploración de los datos.
Obtención de muestras significativas para visualización y reporte.
Extracción de muestras para análisis profundo.

La ejecución de este step:
    - genera un reporte en formato html en la carpeta datasets/parquet
    - almacena samples en formato parquet en la carpeta datasets/samples
    
El objetivo es crear el material para utilizar luego en jupyter notebook.
"""

import os
from typing import Union

import pandas as pd

from tools.eda import automatic_report
from utils.config import PARQUET_PATH, REPORTS_PATH

# pd.options.plotting.backend = "plotly"


def main_research(
    parquet_name: Union[str, None] = None, download_reports: bool = True
) -> None:

    print("  Step 3: Data Research Started  ".center(88, "."), end="\n\n")

    if parquet_name.endswith(".parquet"):
        parquet_name = parquet_name[:-8]
    if parquet_name.endswith("_cleaned"):
        parquet_name = parquet_name[:-8]

    df_raw = pd.read_parquet(os.path.join(PARQUET_PATH, parquet_name + ".parquet"))
    df_cleaned = pd.read_parquet(
        os.path.join(PARQUET_PATH, parquet_name + "_cleaned.parquet")
    )
    df_raw.name = parquet_name
    df_cleaned.name = parquet_name + "_cleaned"

    # Generar reporte del dataset raw en formato html
    if download_reports:
        print("Generating report:")
        automatic_report(df_raw, title=df_raw.name, download=True)
        print(f"Report saved as {df_raw.name} in {REPORTS_PATH}", end="\n\n")
        print("Generating report:")
        automatic_report(df_cleaned, title=df_cleaned.name, download=True)
        print(f"Report saved as {df_cleaned.name} in {REPORTS_PATH}", end="\n\n")

    # Analisis de desbalance de las clases

    # Generar samples de los datos para balancear las clases

    # descargar samples en formato parquet en el directorio de Samples

    print("  Step 3: Data Research Completed  ".center(88, "."), end="\n\n")
