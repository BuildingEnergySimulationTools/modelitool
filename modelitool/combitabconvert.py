import datetime as dt

import pandas as pd


def seconds_to_datetime(index_second, ref_year):
    since = dt.datetime(ref_year, 1, 1, tzinfo=dt.timezone.utc)
    diff_seconds = index_second + since.timestamp()
    return pd.DatetimeIndex(pd.to_datetime(diff_seconds, unit="s"))


def datetime_to_seconds(index_datetime):
    time_start = dt.datetime(index_datetime[0].year, 1, 1, tzinfo=dt.timezone.utc)
    new_index = index_datetime.to_frame().diff().squeeze()
    new_index.iloc[0] = dt.timedelta(
        seconds=index_datetime[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return list(pd.Series(sec_dt).cumsum())


def get_dymo_time_index(df):
    """
    Return a list containing seconds since the beginning of the Year
    Only use UTC datetime index
    """
    time_start = dt.datetime(df.index[0].year, 1, 1, tzinfo=dt.timezone.utc)
    new_index = df.index.to_frame().diff().squeeze()
    new_index.iloc[0] = dt.timedelta(
        seconds=df.index[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return list(pd.Series(sec_dt).cumsum())


def df_to_combitimetable(df, filename):
    """
    Write a text file compatible with modelica Combitimetables object from a
    Pandas DataFrame with a DatetimeIndex. DataFrames with non monotonically increasing
    datetime index will raise a ValueError to prevent bugs when file is used in
    Modelica.
    @param df: DataFrame with DatetimeIndex
    @param filename: string or Path to the output file
    @return: None
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be an instance of pandas DataFrame. Got {type(df)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"DataFrame index must be an instance of DatetimeIndex. " f"Got {type(df)}"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "df DateTimeIndex is not monotonically increasing, this will"
            "cause Modelica to crash."
        )

    df = df.copy()
    with open(filename, "w") as file:
        file.write("#1 \n")
        line = ""
        line += f"double table1({df.shape[0]}, {df.shape[1] + 1})\n"
        line += "\t# Time (s)"
        for i, col in enumerate(df.columns):
            line += f"\t({i + 1}){col}"
        file.write(f"{line} \n")

        df.index = datetime_to_seconds(df.index)

        file.write(df.to_csv(header=False, sep="\t", lineterminator="\n"))
