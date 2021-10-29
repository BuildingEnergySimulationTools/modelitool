import numpy as np
import pandas as pd

from modelitool.measure import MeasuredDats


class TestMeasuredDats:
    def test_minmax_corr(self):
        time_index = pd.date_range(
            "2021-01-01 00:00:00",
            freq="H",
            periods=3
        )

        df = pd.DataFrame(
            {"dumb_column": [-1, 5, 11]},
            index=time_index
        )

        ref = pd.DataFrame(
            {"dumb_column": [np.nan, 5, np.nan]},
            index=time_index
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={},
            corr_dict={},
        )

        tested_obj._minmax_corr(
            "dumb_column",
            upper=10,
            lower=0
        )

        assert ref.equals(tested_obj.corrected_data)

    def test_derivative_corr(self):
        time_index = pd.date_range(
            "2021-01-01 00:00:00",
            freq="H",
            periods=8
        )

        df = pd.DataFrame(
            {"dumb_column": [5, 5.1, 5.1, 6, 7, 22, 6, 5]},
            index=time_index
        )

        ref = pd.DataFrame(
            {"dumb_column": [5, 5.1, np.nan, 6, 7, np.nan, 6, 5]},
            index=time_index
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={},
            corr_dict={},
        )

        lower = 0  # [°C/s]
        upper = 0.004  # [°C/s]

        tested_obj._derivative_corr("dumb_column", upper, lower)

        assert ref.equals(tested_obj.corrected_data)

    def test_ffill(self):
        df = pd.DataFrame(
            {"dumb_column": [2.0, np.nan]},
            index=[0, 1]
        )

        ref = pd.DataFrame(
            {"dumb_column": [2.0, 2.0]},
            index=[0, 1]
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={},
            corr_dict={},
        )

        tested_obj.ffill("dumb_column")

        assert ref.equals(tested_obj.corrected_data)

    def test_bfill(self):
        df = pd.DataFrame(
            {"dumb_column": [np.nan, 2.0]},
            index=[0, 1]
        )

        ref = pd.DataFrame(
            {"dumb_column": [2.0, 2.0]},
            index=[0, 1]
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={},
            corr_dict={},
        )

        tested_obj.bfill("dumb_column")

        assert ref.equals(tested_obj.corrected_data)
