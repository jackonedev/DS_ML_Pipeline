"""
Primer Step en el pipeline de DS

Este modulo se encarga de la lectura de los datos, ya sea de un archivo o de una base de datos.
Hace un relevamiento en la consistencia de los tipos de datos, principalmente la existencia de fechas.
Actualiza los tipos de datos si es necesario.
Descarga una copia en formato parquet.
"""

from utils.config import PROJECT_ROOT
