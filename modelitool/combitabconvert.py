import datetime as dt
import pandas as pd


def seconds_to_datetime(index_second, ref_year):
    since = dt.datetime(ref_year, 1, 1, tzinfo=dt.timezone.utc)
    diff_seconds = index_second + since.timestamp()
    return pd.DatetimeIndex(pd.to_datetime(diff_seconds, unit='s'))


def datetime_to_seconds(index_datetime):
    time_start = dt.datetime(
        index_datetime[0].year, 1, 1, tzinfo=dt.timezone.utc)
    new_index = index_datetime.to_frame().diff().squeeze()
    new_index[0] = dt.timedelta(
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
    new_index[0] = dt.timedelta(
        seconds=df.index[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return list(pd.Series(sec_dt).cumsum())


def df_to_combitimetable(df, filename):
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

        file.write(df.to_csv(
            header=False,
            sep='\t',
            line_terminator='\n'
        ))
