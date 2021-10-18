from datetime import timedelta
import pandas as pd
from dateutil import parser


def get_dymo_time_index(df):
    """
    Return a list containing seconds since the beginning of the Year
    Only use UTC datetime index
    """
    time_start = parser.parse(
        f"{df.index[0].year}-01-01 00:00:00+00:00",
    )
    new_index = df.index.to_frame().diff().squeeze()
    new_index[0] = timedelta(
        seconds=df.index[0].timestamp() - time_start.timestamp()
    )
    sec_dt = [elmt.total_seconds() for elmt in new_index]
    return list(pd.Series(sec_dt).cumsum())


def df_to_combitimetable(df, filename):
    df = df.copy()
    file = open(filename, "w")
    file.write("#1 \n")
    line = ""
    line += f"double table1({df.shape[0]}, {df.shape[1] + 1})"
    line += "\t# Time (s)"
    for i, col in enumerate(df.columns):
        line += f"\t({i}){col}"
    file.write(f"{line} \n")

    df.index = get_dymo_time_index(df)

    file.write(df.to_csv(
        header=False,
        sep='\t',
        line_terminator='\n'
    ))

    file.close()
