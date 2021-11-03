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

    def test_remove_anomalies(self):
        time_index = pd.date_range(
            "2021-01-01 00:00:00",
            freq="H",
            periods=11
        )

        df = pd.DataFrame(
            {
                "dumb_column": [-1, 5, 100, 5, 5.1, 5.1, 6, 7, 22, 6, 5],
                "dumb_column2": [
                    -10, 50, 1000, 50, 50.1, 50.1, 60, 70, 220, 60, 50
                ]
            },
            index=time_index
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [
                    np.nan, 5, np.nan, 5, 5.1, np.nan, 6, 7, np.nan, 6, 5
                ],
                "dumb_column2": [
                    np.nan, 50, np.nan, 50, 50.1, np.nan, 60, 70, np.nan, 60, 50
                ]
            },
            index=time_index
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
            },
            corr_dict={
                "col_1": {
                    "minmax": {
                        "upper": 50,
                        "lower": 0
                    },
                    "derivative": {
                        "lower_rate": 0,
                        "upper_rate": 0.004
                    }
                },
                "col_2": {
                    "minmax": {
                        "upper": 500,
                        "lower": 0
                    },
                    "derivative": {
                        "lower_rate": 0,
                        "upper_rate": 0.04
                    }
                }
            }
        )

        tested_obj.remove_anomalies()

        assert ref.equals(tested_obj.corrected_data)

    def test_fill_nan(self):
        time_index = pd.date_range(
            "2021-01-01 00:00:00",
            freq="H",
            periods=5
        )

        df = pd.DataFrame(
            {
                "dumb_column": [
                    np.nan, 5, np.nan, 7, np.nan
                ],
                "dumb_column2": [
                    np.nan, 5, np.nan, 7, np.nan
                ],
                "dumb_column3": [
                    np.nan, 5, np.nan, 7, np.nan
                ],
            },
            index=time_index
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [
                    5.0, 5.0, 6.0, 7.0, 7.0
                ],
                "dumb_column2": [
                    5.0, 5.0, 7.0, 7.0, 7.0
                ],
                "dumb_column3": [
                    5.0, 5.0, 5.0, 7.0, 7.0
                ],
            },
            index=time_index
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
                "col_3": ["dumb_column3"],
            },
            corr_dict={
                "col_1": {
                    "fill_nan": [
                        "linear_interpolation",
                        "bfill",
                        "ffill"
                    ]
                },
                "col_2": {
                    "fill_nan": [
                        "bfill",
                        "ffill"
                    ]
                },
                "col_3": {
                    "fill_nan": [
                        "ffill",
                        "bfill"
                    ]
                }
            }
        )

        tested_obj.fill_nan()

        print(tested_obj.corrected_data)

        assert ref.equals(tested_obj.corrected_data)

    def test_resample(self):
        time_index_df = pd.date_range(
            "2021-01-01 00:00:00",
            freq="30T",
            periods=4
        )

        df = pd.DataFrame(
            {
                "dumb_column": [
                    5.0, 5.0, 6.0, 6.0
                ],
                "dumb_column2": [
                    5.0, 5.0, 6.0, 6.0
                ],
            },
            index=time_index_df
        )

        time_index_res = pd.date_range(
            "2021-01-01 00:00:00",
            freq="H",
            periods=2
        )

        ref = pd.DataFrame(
            {
                "dumb_column": [
                    5.0, 6.0
                ],
                "dumb_column2": [
                    10.0, 12.0
                ],
            },
            index=time_index_res
        )

        tested_obj = MeasuredDats(
            data=df,
            data_type_dict={
                "col_1": ["dumb_column"],
                "col_2": ["dumb_column2"],
            },
            corr_dict={
                "col_1": {
                    "resample": np.mean
                },
                "col_2": {
                    "resample": np.sum
                },
            }
        )

        tested_obj.resample("H")

        assert ref.equals(tested_obj.corrected_data)