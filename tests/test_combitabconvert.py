import pandas as pd
from modelitool.combitabconvert import get_dymo_time_index
from modelitool.combitabconvert import df_to_combitimetable


class TestCombitabconvert:
    def test_get_dymo_time_index(self):
        time_index = pd.date_range(
            "2021-01-01 01:00:00",
            freq="H",
            periods=3
        )
        df = pd.DataFrame(
            {"dumb_column": [0] * time_index.shape[0]},
            index=time_index
        )
        assert get_dymo_time_index(df) == [3600.0, 7200.0, 10800.0]

    def test_df_to_combitimetable(self, tmpdir):
        time_index = pd.date_range(
                    "2021-01-01 01:00:00",
                    freq="H",
                    periods=3
                )
        df = pd.DataFrame(
                    {"dumb_column": [0] * time_index.shape[0],
                     "dumb_column2": [1] * time_index.shape[0]},
                    index=time_index
                )

        ref = (
            '#1 \n'
            'double table1(3, 3)\n'
            '\t# Time (s)\t(1)dumb_column\t(2)dumb_column2 \n'
            '3600.0\t0\t1\n'
            '7200.0\t0\t1\n'
            '10800.0\t0\t1\n'
        )

        df_to_combitimetable(df, tmpdir / "test.txt")

        with open(tmpdir / "test.txt", 'r') as file:
            contents = file.read()

        assert contents == ref
