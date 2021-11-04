import numpy as np
import pandas as pd
from .combitabconvert import df_to_combitimetable


# TODO Create auto_correct function

def missing_values_dict(df):
    return {
        "Number_of_missing": df.count(),
        "Percent_of_missing": (1 - df.count() / df.shape[0]) * 100
    }


class MeasuredDats:
    def __init__(self, data, data_type_dict, corr_dict):
        self.data = data.copy()
        self.data_type_dict = data_type_dict
        self.corr_dict = corr_dict
        self.corrected_data = data.copy()
        self.correction_journal = {
            "Entries": data.shape[0],
            "Init": missing_values_dict(data)
        }

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
        self.correction_journal["remove_anomalies"] = missing_values_dict(
            self.corrected_data
        )

    def fill_nan(self):
        for data_type, cols in self.data_type_dict.items():
            function_map = {
                "linear_interpolation": self._linear_interpolation,
                "bfill": self._bfill,
                "ffill": self._ffill
            }

            for func in self.corr_dict[data_type]["fill_nan"]:
                function_map[func](cols)

        self.correction_journal["fill_nan"] = missing_values_dict(
            self.corrected_data
        )

    def resample(self, timestep=None):
        if not timestep:
            timestep = self._auto_timestep()

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

    def _auto_timestep(self):
        return self.corrected_data.index.to_frame().diff().mean()[0]

    def generate_combitimetable_input(self, file_path, corrected_data=True):
        if corrected_data:
            df_to_combitimetable(self.corrected_data, file_path)
        else:
            df_to_combitimetable(self.data, file_path)
