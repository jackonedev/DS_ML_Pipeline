from functools import wraps

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
