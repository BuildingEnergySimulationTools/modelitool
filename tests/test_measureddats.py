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
            data_type={},
            corr_dict={},
        )

        tested_obj._minmax_corr(
            "dumb_column",
            upper=10,
            lower=0
        )

        assert ref.equals(tested_obj.corrected_data)
