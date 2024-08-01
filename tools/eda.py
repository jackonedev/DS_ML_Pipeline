import os

import pandas as pd
from ydata_profiling import ProfileReport


def data_info(data: pd.DataFrame, sort=False) -> pd.DataFrame:
    """
    Function to describe the variables of a DataFrame
    Analogous to the .describe() or .info() pandas methods
    Focused on columns overall information
    """
    df = pd.DataFrame(pd.Series(data.columns))
    df.columns = ["columna"]
    df["NaNs"] = data.isna().sum().values
    df["pct_nan"] = round(df["NaNs"] / data.shape[0] * 100, 1)
    df["dtype"] = data.dtypes.values
    df["count"] = data.count().values
    df["count_unique"] = [
        len(data[elemento].value_counts()) for elemento in data.columns
    ]
    df["pct_unique"] = (df["count_unique"].values / data.shape[0] * 100).round(1)
    df = df.reset_index(drop=False)
    if sort:
        df = df.sort_values(by=["dtype", "count_unique"])
        df = df.reset_index(drop=True)
    return df


def automatic_report(df: pd.DataFrame, title: str, download: bool = False) -> dict:
    # el parametro download es para descargar el reporte en el disco local
    # el output_file va a ser el df.name, si no existe se asigna un nombre por default
    # TODO: crear un directory "reports" y configurarlo en utils.config
    profile = ProfileReport(df, title=title)

    # pylint: disable=W0101
    if download:
        raise NotImplementedError("Download option is not implemented yet")
        output_file = df.name if df.name else "report.html"
        # TODO: terminar de configuar el path de descarga con utils.config
        output_file = os.path.abspath(output_file)
        profile.to_file(output_file)

    # return profile.to_json()
    return profile


def categorize_columns(X: pd.DataFrame, cat_id_threshold: int) -> dict:
    """
    Categorize columns in a DataFrame into binary, categorical, numerical, id and rest columns.

    Args:
    X (pd.DataFrame): DataFrame to categorize columns.
    cat_id_threshold (int): the number of unique values a column must have to be considered as an id.

    Returns:
    dict: Dictionary with keys as column categories and values as column names.
    """
    numerical_columns = X.select_dtypes(include="number").columns

    numerical_columns = X.select_dtypes(include="number").columns
    # TODO: Si es TimeDelta corresponde convertirlo a type int

    categories_count = (
        X.loc[:, (X.dtypes == "object").values]
        .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
        .sum()
    )

    binary_columns = categories_count[categories_count == 2].index.to_list()
    categorical_columns = categories_count[
        (2 < categories_count) & (categories_count < cat_id_threshold)
    ].index.to_list()
    id_columns = categories_count[
        (cat_id_threshold <= categories_count)
    ].index.to_list()
    # TODO: datetime_columns = Error de implementacion en la detección del datetime

    rest_columns = X.columns.difference(binary_columns + categorical_columns)
    rest_columns = rest_columns[~rest_columns.isin(numerical_columns)]
    rest_columns = rest_columns[~rest_columns.isin(id_columns)]

    return {
        "id_columns": id_columns,
        "binary_columns": binary_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "rest_columns": rest_columns,
    }
