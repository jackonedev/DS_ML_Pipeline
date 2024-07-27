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

        >>> pd.Series([...]).apply(format_date)
        pd.Series([...])
    """
    dt = pendulum.parse(date_str)
    pendulum.set_locale(language)
    return dt.format("dddd DD MMMM YYYY")
