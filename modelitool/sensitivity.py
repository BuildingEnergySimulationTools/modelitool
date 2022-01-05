from SALib.sample import fast_sampler
from SALib.sample import saltelli
from SALib.sample import morris as morris_sampler

from SALib.analyze import fast
from SALib.analyze import morris
from SALib.analyze import sobol

from modelitool.simulate import run_batch

import plotly.graph_objects as go

import numpy as np
import pandas as pd
import datetime as dt

import time


def check_arguments(res, param, result):
    if result not in res:
        raise ValueError(f"{result} not found in results")

    if res[result].shape[0] != len(param):
        raise ValueError("Parameters name and index length mismatch")


def modelitool_to_salib_problem(modelitool_problem):
    return {
        'num_vars': len(modelitool_problem),
        'names': list(modelitool_problem.keys()),
        'bounds': list(modelitool_problem.values())
    }


class SAnalysis:
    def __init__(
            self,
            simulator,
            sensitivity_method,
            parameters_config,
    ):
        self.meth_samp_map = {
            "FAST": {
                "method": fast,
                "sampling": fast_sampler,
            },
            "Morris": {
                "method": morris,
                "sampling": morris_sampler
            },
            "Sobol": {
                "method": sobol,
                "sampling": saltelli,
            }
        }
        self.simulator = simulator
        self.parameters_config = parameters_config
        if sensitivity_method not in self.meth_samp_map.keys():
            raise ValueError('Specified sensitivity method is not valid')
        else:
            self._sensitivity_method = sensitivity_method

        self.salib_problem = modelitool_to_salib_problem(
            self.parameters_config)

        self.sample = np.array([])
        self.simulation_results = np.array([])
        self.sensitivity_results = None

    @property
    def simulator_outputs(self):
        return self.simulator.output_list

    def draw_sample(self, n, arguments=None):
        if arguments is None:
            arguments = {}

        sampler = self.meth_samp_map[self._sensitivity_method]['sampling']
        self.sample = sampler.sample(
            N=n,
            problem=self.salib_problem,
            **arguments
        )

    def run_simulations(self, verbose_step=10):
        if self.sample.size == 0:
            raise ValueError(
                'No sample available. Generate sample using draw_sample()'
            )

        self.simulation_results = run_batch(
            simulator=self.simulator,
            param_name_list=list(self.parameters_config.keys()),
            sample=self.sample,
            verbose_step=verbose_step
        )

    def get_indicator_from_simulation_results(
            self, aggregation_method, indicator, ref=None):
        ind_res = np.zeros(self.simulation_results.shape[0])
        for idx, res in enumerate(self.simulation_results[
                                  :, :,
                                  self.simulator_outputs.index(indicator)]):
            if ref is None:
                ind_res[idx] = aggregation_method(res)
            else:
                ind_res[idx] = aggregation_method(res, ref)
        return ind_res

    def analyze(
            self,
            indicator,
            aggregation_method,
            reference=None,
            arguments=None):

        if arguments is None:
            arguments = {}

        if indicator not in self.simulator_outputs:
            raise ValueError('Specified indicator not in computed outputs')

        y_array = self.get_indicator_from_simulation_results(
            aggregation_method, indicator, reference)

        analyser = self.meth_samp_map[self._sensitivity_method]["method"]

        self.sensitivity_results = analyser.analyze(
            problem=self.salib_problem,
            Y=y_array,
            **arguments
        )

    def plot(self, kind="bar", arguments=None):
        if kind == "bar":
            if self.sensitivity_results is None:
                raise ValueError("No result to plot")

            if self._sensitivity_method == "Sobol":
                plot_sobol_st_bar(
                    self.sensitivity_results,
                    self.parameters_config.keys()
                )

        if kind == "parallel":
            if arguments is None:
                raise ValueError("Please provide list of dict to compute "
                                 "indicators values")

            ind_dict = {}
            for ind in arguments["indicator_dict_list"]:
                ind_dict[ind[
                    "name"]] = self.get_indicator_from_simulation_results(
                    **{
                        k: ind[k]
                        for k in ind.keys()
                        if k in ["aggregation_method", "indicator", "ref"]
                    }
                )

            param_dict = {
                par: values
                for par, values in zip(
                    self.parameters_config, self.sample.T
                )
            }

            if "plot_options" not in arguments.keys():
                options = {"colorby": list(ind_dict.keys())[0]}
            else:
                options = arguments["plot_options"]

            plot_parcoord(
                data_dict={**param_dict, **ind_dict},
                **options
            )

        if kind == "sample":
            if arguments is None:
                raise ValueError("Please specify at least model output name")

            options = {key: val for key, val in arguments.items()
                       if key != 'indicator'}

            plot_sample(
                sample_res=self.simulation_results[
                    :, :, self.simulator_outputs.index(arguments['indicator'])
                ],
                **options
            )


def plot_parcoord(data_dict, colorby=None, colorscale='Electric'):
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=data_dict[colorby],
            colorscale=colorscale,
            showscale=True,
            cmin=data_dict[colorby].min(),
            cmax=data_dict[colorby].max()
        ),
        dimensions=[
            {
                "range": [data_dict[par].min(),
                          data_dict[par].max()],
                "label": par,
                "values": data_dict[par]
            }
            for par in data_dict.keys()
        ]
    ))

    fig.show()


def plot_sobol_st_bar(salib_res, param_names):
    check_arguments(salib_res, param_names, "ST")

    sobol_ind = salib_res.to_df()[0]
    sobol_ind.sort_values(by="ST", ascending=True, inplace=True)

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=sobol_ind.index,
        y=sobol_ind.ST,
        name="Sobol Total Indices",
        marker_color='orange',
        error_y=dict(type="data", array=sobol_ind.ST_conf.to_numpy()),
        yaxis="y1"
    ))

    figure.update_layout(
        title='Sobol Total indices',
        xaxis_title='Parameters',
        yaxis_title='Sobol total index value [0-1]'
    )

    figure.show()


def plot_sample(sample_res, ref=None, title=None, y_label=None, x_label=None,
                x_axis=None):

    n_sample = sample_res.shape[0]
    x_to_plot = np.concatenate(
        [np.arange(sample_res.shape[1])]*n_sample)
    y_to_plot = sample_res.flatten()

    if isinstance(ref, pd.Series) or isinstance(ref, pd.DataFrame):
        x_to_plot = pd.concat([pd.Series(ref.index)] * n_sample)

    elif x_axis is not None:
        if isinstance(x_axis, np.array) or isinstance(x_axis, list):
            x_to_plot = np.concatenate([x_axis] * n_sample)

        elif isinstance(x_axis, pd.DatetimeIndex):
            x_to_plot = pd.concat([pd.Series(x_axis)] * n_sample)

        elif isinstance(x_axis, pd.Series):
            x_to_plot = pd.concat([x_axis] * n_sample)

        elif isinstance(x_axis, pd.DataFrame):
            x_to_plot = pd.concat([x_axis.squeeze()] * n_sample)

        else:
            raise ValueError("x_axis has a wrong format. Please provide"
                             "list, np.array, pd.Series, pd.DataFrame")

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            name="Sample",
            mode="markers",
            x=x_to_plot,
            y=y_to_plot,
            marker=dict(
                color='rgba(135, 135, 135, 0.02)',
            )
        )
    )

    if ref is not None:
        fig.add_trace(
            go.Scattergl(
                name="Reference",
                mode='lines',
                x=ref.index,
                y=ref,
                line=dict(
                    color='crimson',
                    width=2
                )
            )
        )

    if title is not None:
        fig.update_layout(title=title)
    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)

    fig.show()
