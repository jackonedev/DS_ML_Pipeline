from functools import wraps

import numpy as np
import pandas as pd
import pendulum


def format_date(date_str: str, language: str = "es") -> str:
    """
    Formats a date string into a humanized format based on the given language.

    Parameters:
        date_str (str): The date string to format.
        language (str, optional): The language code for the desired format. Defaults to 'es'.

    Returns:
        str: The formatted date string.

    Example:
        >>> format_date('2022-01-01', 'en')
        'Saturday 01 January 2022'

        >>> pd.Series([ <str>, <str>, ... ]).apply(format_date)
        pd.Series([ <str>, <str>, ... ])

        >>> pd.Series([ <datetime>, <datetime>, ... ]).apply(str(x.date())).apply(format_date)
        pd.Series([ <str>, <str>, ... ])
    """

    # pylint: disable=W0718
    try:
        dt = pendulum.parse(date_str)
    except Exception:
        # The code returns a ParseError and is not a built-in exception
        return date_str

    pendulum.set_locale(language)
    return dt.format("dddd DD MMMM YYYY")


def remove_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return pd.Timedelta(days=result.days)

    return wrapper


@remove_time
def round_timedelta(td: pd.Timedelta) -> pd.Timedelta:
    """
    Rounds a timedelta to the nearest hour, splitting the day into 12-hour intervals.

    Parameters:
        td (pd.Timedelta): The timedelta to round.

    Returns:
        pd.Timedelta: The rounded timedelta.
    """
    total_seconds = td.total_seconds()
    # Each 12 hours is 12 * 3600 seconds
    interval = 12 * 3600
    rounded_seconds = round(total_seconds / interval) * interval
    return pd.Timedelta(seconds=rounded_seconds)


def extract_date_features(
    df: pd.DataFrame, col: str, include_col_name: bool = True
) -> pd.DataFrame:
    """
    Extracts various date features from a specified column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the date column.
        col (str): The name of the date column.
        include_col_name (bool, optional): Whether to include the column name as a prefix in the feature names.
                                            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted date features.
    """
    col_name = f"{col}_" if include_col_name else ""
    result = pd.DataFrame()
    result[f"{col_name}year"] = df[col].dt.year
    result[f"{col_name}month"] = df[col].dt.month
    result[f"{col_name}day"] = df[col].dt.day
    result[f"{col_name}hour"] = df[col].dt.hour
    result[f"{col_name}dayofweek"] = df[col].dt.dayofweek
    result[f"{col_name}month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
    result[f"{col_name}month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
    result[f"{col_name}day_sin"] = np.sin(2 * np.pi * df[col].dt.day / 31)
    result[f"{col_name}day_cos"] = np.cos(2 * np.pi * df[col].dt.day / 31)
    return result
