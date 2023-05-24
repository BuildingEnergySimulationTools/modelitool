import pandas as pd

from modelitool.combitabconvert import df_to_combitimetable
from modelitool.combitabconvert import seconds_to_datetime
from modelitool.combitabconvert import datetime_to_seconds

from datetime import timedelta

import pytest


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
        assert datetime_to_seconds(df.index) == [3600.0, 7200.0, 10800.0]

    def test_df_to_combitimetable(self, tmpdir):
        with pytest.raises(ValueError):
            df_to_combitimetable([1, 2, 3], tmpdir / "test.txt")

        with pytest.raises(ValueError):
            df_to_combitimetable(
                pd.DataFrame(data=[1, 2, 3], index=[1, 2, 3]), tmpdir / "test.txt"
            )

        with pytest.raises(ValueError):
            df_to_combitimetable(
                pd.DataFrame(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(
                        [
                            "2020-01-01 00:00:00",
                            "2019-12-31 23:00:00",
                            "2020-01-01 01:00:00",
                        ]
                    ),
                ),
                tmpdir / "test.txt",
            )

        time_index = pd.date_range("2021-01-01 01:00:00", freq="H", periods=3)
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

    def test_seconds_to_datetime(self):
        test_index = pd.Series([
            timedelta(seconds=43200).total_seconds(),
            timedelta(seconds=43500).total_seconds()
        ])

        expected_res = pd.to_datetime([
            "2009-01-01 12:00:00",
            "2009-01-01 12:05:00"
        ])

        res = seconds_to_datetime(test_index, 2009)

        pd.testing.assert_index_equal(expected_res, res)
