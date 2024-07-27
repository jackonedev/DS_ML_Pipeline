import pandas as pd


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
