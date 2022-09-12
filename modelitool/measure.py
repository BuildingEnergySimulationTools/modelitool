import numpy as np
import pandas as pd
from .combitabconvert import df_to_combitimetable


def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100
    }


def find_gaps(df_in, cols=None, timestep=None):
    if not cols:
        cols = df_in.columns

    if not timestep:
        timestep = auto_timestep(df_in)

    # Aggregate in a single columns to know overall quality
    df = df_in.copy()
    df = ~df.isnull()
    df["combination"] = df.all(axis=1)

    # Index are added at the beginning and at the end to account for
    # missing values and each side of the dataset
    first_index = df.index[0] - (df.index[1] - df.index[0])
    last_index = df.index[-1] - (df.index[-2] - df.index[-1])

    df.loc[first_index] = np.ones(df.shape[1], dtype=bool)
    df.loc[last_index] = np.ones(df.shape[1], dtype=bool)
    df.sort_index(inplace=True)

    # Compute gaps duration
    res = {}
    for col in list(cols) + ["combination"]:
        time_der = df[col].loc[df[col]].index.to_series().diff()
        res[col] = time_der[time_der > timestep] - timestep

    return res


def gaps_describe(df_in, cols=None, timestep=None):
    res_find_gaps = find_gaps(df_in, cols, timestep)

    return pd.DataFrame(
        {k: val.describe() for k, val in res_find_gaps.items()})


def auto_timestep(df):
    return df.index.to_frame().diff().mean()[0]


class MeasuredDats:
    def __init__(self, data, data_type_dict, corr_dict, gaps_timedelta=None):
        self.data = data.apply(
            pd.to_numeric, args=('coerce',)
        ).copy()
        self.corrected_data = data.apply(
            pd.to_numeric, args=('coerce',)
        ).copy()

        self.data_type_dict = data_type_dict
        self.corr_dict = corr_dict
        self.correction_journal = {
            "Entries": data.shape[0],
            "Init": missing_values_dict(data)
        }
        if gaps_timedelta is None:
            self.gaps_timedelta = auto_timestep(self.data)
        else:
            self.gaps_timedelta = gaps_timedelta

    def auto_correct(self):
        self.remove_anomalies()
        self.fill_nan()
        self.resample()

    def remove_anomalies(self):
        for data_type, cols in self.data_type_dict.items():
            self._minmax_corr(
                cols=cols,
                **self.corr_dict[data_type]["minmax"]
            )

            self._derivative_corr(
                cols=cols,
                **self.corr_dict[data_type]["derivative"]
            )
        self.correction_journal["remove_anomalies"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta)
        }

    def fill_nan(self):
        for data_type, cols in self.data_type_dict.items():
            function_map = {
                "linear_interpolation": self._linear_interpolation,
                "bfill": self._bfill,
                "ffill": self._ffill
            }

            for func in self.corr_dict[data_type]["fill_nan"]:
                function_map[func](cols)

        self.correction_journal["fill_nan"] = {
            "missing_values": missing_values_dict(self.corrected_data),
            "gaps_stats": gaps_describe(
                self.corrected_data, timestep=self.gaps_timedelta)
        }

    def resample(self, timestep=None):
        if not timestep:
            timestep = auto_timestep(self.corrected_data)

        agg_arguments = {}
        for data_type, cols in self.data_type_dict.items():
            for col in cols:
                agg_arguments[col] = self.corr_dict[data_type]["resample"]

        resampled = self.corrected_data.resample(timestep).agg(agg_arguments)
        self.corrected_data = resampled

        self.correction_journal["Resample"] = f"Resampled at {timestep}"

    def _minmax_corr(self, cols, upper, lower):
        df = self.corrected_data.loc[:, cols]
        upper_mask = df > upper
        lower_mask = df < lower
        mask = np.logical_or(upper_mask, lower_mask)
        self.corrected_data[mask] = np.nan

    def _derivative_corr(self, cols, upper_rate, lower_rate):
        df = self.corrected_data.loc[:, cols]
        time_delta = df.index.to_series().diff().dt.total_seconds()
        abs_der = abs(
            df.diff().divide(time_delta, axis=0)
        )
        abs_der_two = abs(
            df.diff(periods=2).divide(time_delta, axis=0)
        )

        mask_constant = abs_der <= lower_rate
        mask_der = abs_der >= upper_rate
        mask_der_two = abs_der_two >= upper_rate

        mask_to_remove = np.logical_and(mask_der, mask_der_two)
        mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

        self.corrected_data[mask_to_remove] = np.nan

    def _linear_interpolation(self, cols):
        self._interpolate(cols, method='linear')

    def _interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter

    def _ffill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="ffill"
        )
        self.corrected_data.loc[:, cols] = filled

    def _bfill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="bfill"
        )
        self.corrected_data.loc[:, cols] = filled

    def generate_combitimetable_input(self, file_path, corrected_data=True):
        if corrected_data:
            df_to_combitimetable(self.corrected_data, file_path)
        else:
            df_to_combitimetable(self.data, file_path)
