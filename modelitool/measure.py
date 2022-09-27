import numpy as np
import pandas as pd
from modelitool.combitabconvert import df_to_combitimetable
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import datetime as dt


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
        res[col] = time_der[time_der > timestep]

    return res


def gaps_describe(df_in, cols=None, timestep=None):
    res_find_gaps = find_gaps(df_in, cols, timestep)

    return pd.DataFrame(
        {k: val.describe() for k, val in res_find_gaps.items()})


def auto_timestep(df):
    return df.index.to_frame().diff().mean()[0]


def add_scatter_and_gaps(figure, series, gap_series, color_rgb, alpha, y_min,
                         y_max):
    figure.add_trace(go.Scattergl(
        x=series.index,
        y=series.to_numpy().flatten(),
        mode='lines+markers',
        name=series.name,
        # line=dict(color=f'rgb{color_rgb}')
    ))

    for t_idx, gap in gap_series.iteritems():
        figure.add_trace(go.Scattergl(
            x=[t_idx - gap, t_idx - gap, t_idx, t_idx],
            y=[y_min, y_max, y_max, y_min],
            mode='none',
            fill='toself',
            showlegend=False,
            fillcolor=f"rgba({color_rgb[0]}, {color_rgb[1]},"
                      f" {color_rgb[2]} , {alpha})",
        ))


class MeasuredDats:
    def __init__(self,
                 data,
                 data_type_dict=None,
                 corr_dict=None,
                 gaps_timedelta=None):

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

        self.resample_func_dict = {
            'mean': np.mean,
            'sum': np.sum
        }

    @property
    def columns(self):
        return self.data.columns

    def auto_correct(self):
        self.remove_anomalies()
        self.fill_nan()
        self.resample()

    def remove_anomalies(self):
        for data_type, cols in self.data_type_dict.items():
            if "minmax" in self.corr_dict[data_type].keys():
                self._minmax_corr(
                    cols=cols,
                    **self.corr_dict[data_type]["minmax"]
                )
            if "derivative" in self.corr_dict[data_type].keys():
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
                key = self.corr_dict[data_type]["resample"]
                agg_arguments[col] = self.resample_func_dict[key]

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

    def get_scaled_data(self, cols=None, scaler=StandardScaler):
        if cols is None:
            cols = self.data.columns

        df_raw = self.data[cols]
        df_corr = self.corrected_data[cols]

        scal = scaler()
        scal.fit(df_raw)
        df_raw[cols] = scal.transform(df_raw[cols])
        df_corr[cols] = scal.transform(df_corr[cols])

        return df_raw, df_corr

    def plot_gaps(
            self,
            cols=None,
            gaps_timestep=dt.timedelta(hours=5),
            y_label=None,
            title="Gaps plot",
            scale_data=False,
            scaler=StandardScaler,
            raw_data=False,
            color_rgb=(243, 132, 48),
            alpha=0.5):

        if cols is None:
            cols = self.columns

        if scale_data:
            raw_scaled, corr_scaled = self.get_scaled_data(cols, scaler)
            if raw_data:
                to_plot = raw_scaled
            else:
                to_plot = corr_scaled
        else:
            if raw_data:
                to_plot = self.data
            else:
                to_plot = self.corrected_data

        if isinstance(to_plot, pd.Series):
            to_plot = to_plot.to_frame()

        y_min = to_plot.min().min()
        y_max = to_plot.max().max()

        fig = go.Figure()

        for col in cols:
            add_scatter_and_gaps(
                figure=fig,
                series=to_plot[col],
                gap_series=find_gaps(
                    df_in=to_plot, cols=[col], timestep=gaps_timestep)[col],
                color_rgb=color_rgb,
                alpha=alpha,
                y_min=y_min,
                y_max=y_max)

        fig.update_layout(dict(
            title=title,
            yaxis_title=y_label
        ))

        fig.show()

    def plot(
            self, cols=None, title="Correction plot",
            scale_data=False, scaler=StandardScaler, plot_raw=False,
            begin=None, end=None):

        if begin is None:
            begin = self.corrected_data.index[0]

        if end is None:
            end = self.corrected_data.index[-1]

        if cols is None:
            cols = self.columns

        if scale_data:
            to_plot_raw, to_plot_corr = self.get_scaled_data(cols, scaler)
            to_plot_raw = to_plot_raw.loc[begin:end, :]
            to_plot_corr = to_plot_corr.loc[begin:end, :]
        else:
            to_plot_raw = self.data.loc[begin:end, cols]
            to_plot_corr = self.corrected_data.loc[begin:end, cols]

        fig = go.Figure()

        ax_dict = {}
        for col in cols:
            for key, name_list in self.data_type_dict.items():
                if col in name_list:
                    ax_dict[col] = key

        ax_map = {cat: f"y{i + 1}"
                  for i, cat in enumerate(set(ax_dict.values()))}
        ax_map[list(ax_map.keys())[0]] = 'y'
        ax_dict = {k: ax_map[ax_dict[k]] for k in ax_dict.keys()}

        for col in cols:
            if plot_raw:
                fig.add_scattergl(
                    x=to_plot_raw.index,
                    y=to_plot_raw[col],
                    name=f"{col}_raw",
                    mode='lines+markers',
                    line=dict(color=f'rgb(216,79,86)'),
                    yaxis=ax_dict[col]
                )

            fig.add_scattergl(
                x=to_plot_corr.index,
                y=to_plot_corr[col],
                name=f"{col}_corrected",
                mode='lines+markers',
                yaxis=ax_dict[col]
            )

        layout_ax_dict = {}
        ax_list = list(ax_map.keys())
        layout_ax_dict["yaxis"] = {"title": ax_list[0]}
        for i, ax in enumerate(ax_list[1:]):
            layout_ax_dict[f"yaxis{i + 2}"] = {
                "title": ax,
                "side": "right"
            }

        fig.update_layout(**layout_ax_dict)

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5),
        )

        fig.update_layout(dict(title=title))

        fig.show()
