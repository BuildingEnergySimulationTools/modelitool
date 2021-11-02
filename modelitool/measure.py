import numpy as np
from .combitabconvert import df_to_combitimetable


# TODO Create auto_correct function
# TODO Create a fill_nan function
# TODO Add resample in some way

class MeasuredDats:
    def __init__(self, data, data_type_dict, corr_dict):
        self.data = data
        self.data_type_dict = data_type_dict
        self.corr_dict = corr_dict
        self.corrected_data = data
        self.correction_journal = []

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
        self.correction_journal.append("remove_anomalies")

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

        print(abs_der)

        mask_to_remove = np.logical_and(mask_der, mask_der_two)
        mask_to_remove = np.logical_or(mask_to_remove, mask_constant)

        self.corrected_data[mask_to_remove] = np.nan

    def interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter
        self.correction_journal.append("interpolate")

    def ffill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="ffill"
        )
        self.corrected_data.loc[:, cols] = filled
        self.correction_journal.append("ffill")

    def bfill(self, cols):
        filled = self.corrected_data.loc[:, cols].fillna(
            method="bfill"
        )
        self.corrected_data.loc[:, cols] = filled
        self.correction_journal.append("bfill")

    def generate_combitimetable_input(self, file_path, corrected_data=True):
        if corrected_data:
            if self.correction_journal:
                df_to_combitimetable(self.corrected_data, file_path)
            else:
                raise ValueError('Cannot compose from corrected_data.\
                    No correction were applied')
        else:
            df_to_combitimetable(self.data, file_path)
