import os
from typing import Dict, List, Tuple

import pandas as pd
from ydata_profiling import ProfileReport

from utils.config import REPORTS_PATH


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
    """
    Create a ProfileReport object from a DataFrame and save it as an HTML file.
    The name of the file is the name of the DataFrame if it has one, otherwise it is "report.html".
    
    Args:
    df (pd.DataFrame): DataFrame to create the report from.
    title (str): Title of the report.
    download (bool): Whether to save the report as an HTML file or not.
    
    Returns:
    ProfileReport: ProfileReport object.
    """
    profile = ProfileReport(df, title=title)

    if download:
        output_file = df.name if df.name else "report.html"
        output_file = output_file if output_file.endswith(".html") else f"{output_file}.html"
        output_file = os.path.join(REPORTS_PATH, output_file)
        profile.to_file(output_file)

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


def change_cols_order(
    df: pd.DataFrame, init_cols: List[str], final_cols: List[str]
) -> pd.DataFrame:
    """
    Reorders the columns of a DataFrame based on the given initial and final column lists.
    Args:
        df (pd.DataFrame): The DataFrame to be reordered.
        init_cols (List[str]): The list of initial columns.
        final_cols (List[str]): The list of final columns.
    Returns:
        pd.DataFrame: The reordered DataFrame.
    """
    assert all(col in df.columns for col in init_cols + final_cols), "Column not found"

    order = (
        init_cols
        + [col for col in df.columns if col not in init_cols + final_cols]
        + final_cols
    )
    return df.reindex(columns=order)
