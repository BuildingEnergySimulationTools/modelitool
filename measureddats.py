import numpy as np


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
                self.minmax_corr(
                    cols=cols,
                    **self.corr_dict[type]["minmax"]
                )

            if "derivative" in corrections:
                self.derivative_corr(
                    cols=cols,
                    **self.corr_dict[type]["derivative"]
                )

            if "interpolate" in corrections:
                self.interpolate(
                    cols=cols,
                    **self.corr_dict[type]["interpolate"]
                )

            if "ffill" in corrections:
                filled = self.corrected_data.loc[:, cols].fillna("ffill")
                self.corrected_data.loc[:, cols] = filled
                self.applied_corr.append("ffill")

            if "bfill" in corrections:
                filled = self.corrected_data.loc[:, cols].fillna("bfill")
                self.corrected_data.loc[:, cols] = filled
                self.applied_corr.append("bfill")

    def minmax_corr(self, cols, upper, lower):
        df = self.corrected_data.loc[:, cols]
        upper_mask = df > upper
        lower_mask = df < lower
        mask = np.logical_or(upper_mask, lower_mask)
        self.corrected_data[mask] = np.nan
        self.applied_corr.append("minmax")

    def derivative_corr(self, cols, rate):
        df = self.corrected_data.loc[:, cols]
        der = df.diff()
        mask = abs(der) > rate
        self.corrected_data[mask] = np.nan
        self.applied_corr.append("derivative")

    def interpolate(self, cols, method):
        inter = self.corrected_data.loc[:, cols].interpolate(method=method)
        self.corrected_data.loc[:, cols] = inter
        self.applied_corr.append("interpolate")