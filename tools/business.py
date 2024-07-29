import pandas as pd


def eliminar_cursada(df: pd.DataFrame, df_outlier: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el dataset original, y aquel solo con outliers.
    Extrae 'user_uuid', 'course_uuid', 'course_name', y 'legajo'
    de cada registro de los outliers, busca toda las particiones
    correspondientes, extrae los indices, y los remueve del
    dataset original.
    """

    outlier_reg = [
        dict(zip(df_outlier.columns[-4:], val))
        for val in df_outlier.iloc[:, -4:].values
    ]

    # Handling Edge Cases
    # Manejando posibles alteraciones en el dataset
    expected_keys = ["user_uuid", "course_uuid", "course_name", "legajo"]
    assert (
        list(outlier_reg[0].keys()) == expected_keys
    ), f"Las claves no coinciden. Se esperaban {expected_keys} pero se obtuvieron {list(outlier_reg[0].keys())}"

    # Obtener los índices de los registros a eliminar
    indexes = []
    for reg in outlier_reg:
        indexes.append(
            df[(df[list(reg.keys())].eq(list(reg.values())).all(axis=1))].index
        )
    indexes = [item for sublist in indexes for item in sublist]

    return df.drop(indexes)
