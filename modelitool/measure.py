import numpy as np
from .combitabconvert import df_to_combitimetable


class MeasuredDats:
    def __init__(self, data, data_type, corr_dict):
        self.data = data
        self.data_type = data_type
        self.corr_dict = corr_dict
        self.corrected_data = data
        self.applied_corr = []

    def get_corrected(
            self,
            corrections=[
                "minmax",
                "derivative",
                "interpolate",
                "ffill",
                "bfill"
            ]
    ):
        self.corrected_data = self.data.copy()
        self.applied_corr.clear()

        for type, cols in self.data_type.items():
            if "minmax" in corrections:
                self._minmax_corr(
                    cols=cols,
                    **self.corr_dict[type]["minmax"]
                )

            if "derivative" in corrections:
                self._derivative_corr(
                    cols=cols,
                    **self.corr_dict[type]["derivative"]
                )

            if "interpolate" in corrections:
                self._interpolate(
                    cols=cols,
                    **self.corr_dict[type]["interpolate"]
                )

            if "ffill" in corrections:
                filled = self.corrected_data.loc[:, cols].fillna(
                    method="ffill"
                )
                self.corrected_data.loc[:, cols] = filled

            if "bfill" in corrections:
                filled = self.corrected_data.loc[:, cols].fillna(
                    method="bfill"
                )
                self.corrected_data.loc[:, cols] = filled

    def _minmax_corr(self, cols, upper, lower):
        df = self.corrected_data.loc[:, cols]
        upper_mask = df > upper
        lower_mask = df < lower
        mask = np.logical_or(upper_mask, lower_mask)
        self.corrected_data[mask] = np.nan

    def _derivative_corr(self, cols, upper_rate, lower_rate):
        df = self.corrected_data.loc[:, cols]
        time_delta = df.index.to_series().diff().dt.total_seconds()
        abs_der = abs(df.diff() / time_delta)
        abs_der_two = abs(df.diff(periods=2) / time_delta)

        mask_constant = abs_der <= lower_rate
        mask_der = abs_der >= upper_rate
        mask_der_two = abs_der_two >= upper_rate

        mask_to_remove = np.logical_and(mask_der, mask_der_two)
        mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

        self.corrected_data[mask_to_remove] = np.nan

    def _interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter
        self.applied_corr.append("interpolate")

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
            if self.applied_corr:
                df_to_combitimetable(self.corrected_data, file_path)
            else:
                raise ValueError('Connot compose from corrected_data.\
                    No correction were applied')
        else:
            df_to_combitimetable(self.data, file_path)