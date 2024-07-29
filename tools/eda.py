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
