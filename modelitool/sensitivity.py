from SALib.sample import fast_sampler
from SALib.sample import saltelli
from SALib.sample import morris as morris_sampler

from SALib.analyze import fast
from SALib.analyze import morris
from SALib.analyze import sobol

from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior

import plotly.graph_objects as go

import numpy as np
import pandas as pd
import datetime as dt
import warnings
import time

master_bar, progress_bar = force_console_behavior()


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

        self.simulation_time_index = []
        self.sample = np.array([])
        self.simulation_results = list()
        self.sensitivity_results = None
        self.sensitivity_dynamic_results = {}

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

        self.simulation_results = list()
        simu_list = []

        for samp in self.sample:
            simu_list.append({
                param: val for param, val in zip(
                    self.parameters_config.keys(), samp
                )
            })

        # A bit dirty but perform first simulation to get
        # output shape and simulation time estimation
        t1 = time.time()

        self.simulator.set_param_dict(simu_list[0])
        self.simulator.simulate()
        results = self.simulator.get_results()

        t2 = time.time()
        sim_duration = dt.timedelta(seconds=t2 - t1)

        self.simulation_results.append(results)

        # Run remaining run_simulations
        for idx, sim in enumerate(simu_list[1:]):
            if not idx % verbose_step:
                print(f"Running simulation {idx + 2}/{len(simu_list)}")
                remaining_sec = sim_duration * (len(simu_list) - idx + 1)
                rem_days = remaining_sec.days
                rem_hours, rem = divmod(remaining_sec.seconds, 3600)
                rem_minutes, rem_seconds = divmod(rem, 60)
                print(
                    f"Remaining: {rem_days} days {rem_hours}h{rem_minutes}′{rem_seconds}″"
                )

            self.simulator.set_param_dict(sim)
            self.simulator.simulate()
            results = self.simulator.get_results()
            self.simulation_results.append(results)

    def _compute_aggregated(
            self, aggregation_method, indicator, ref=None, freq=None):

        aggregated_list = []

        if freq is None:
            for res in self.simulation_results:
                if ref is None:
                    aggregated_list.append(aggregation_method(res[indicator]))
                else:
                    aggregated_list.append(
                        aggregation_method(res[indicator], ref))
        else:
            grouper = pd.Grouper(freq=freq)
            prog_bar = progress_bar(range(len(self.simulation_results)))

            for mb, res in zip(prog_bar, self.simulation_results):
                tempos = pd.Series()
                simu_group = res[indicator].groupby(grouper)
                prog_bar.comment = 'Aggregation'

                if ref is not None:
                    ref_group = ref.groupby(grouper)
                    for simu_gr, ref_gr in zip(simu_group, ref_group):
                        tempos[simu_gr[0]] = aggregation_method(
                            ref_gr[1], simu_gr[1])
                else:
                    for simu_gr in simu_group:
                        tempos[simu_gr[0]] = aggregation_method(simu_gr[1])

                aggregated_list.append(tempos)

        return aggregated_list

    def analyze(
            self,
            indicator,
            aggregation_method,
            reference=None,
            freq=None,
            arguments=None):

        if arguments is None:
            arguments = {}

        if indicator not in self.simulator_outputs:
            raise ValueError('Specified indicator not in computed outputs')

        analyser = self.meth_samp_map[self._sensitivity_method]["method"]

        if freq is None:
            y_array = np.array(self._compute_aggregated(
                aggregation_method, indicator, reference))

            self.sensitivity_results = analyser.analyze(
                problem=self.salib_problem,
                Y=y_array,
                **arguments
            )
        else:
            agg_list = self._compute_aggregated(
                aggregation_method=aggregation_method,
                indicator=indicator,
                ref=reference,
                freq=freq,
            )

            index = agg_list[0].index
            numpy_res = np.array(agg_list).T
            prog_bar = progress_bar(range(index.shape[0]))

            for bar, idx, res in zip(prog_bar, index, numpy_res):
                prog_bar.comment = 'Dynamic index'
                self.sensitivity_dynamic_results[idx] = analyser.analyze(
                    problem=self.salib_problem,
                    Y=res,
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
                arg_dict = {
                    key: ind[key]
                    for key in ind.keys()
                    if key in ["aggregation_method", "indicator", "ref"]
                }

                ind_dict[ind["name"]] = np.array(self._compute_aggregated(
                    **arg_dict))

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
                raise ValueError("Please specify at least model output"
                                 " name as 'indicator")

            options = {key: val for key, val in arguments.items()
                       if key != 'indicator'}

            plot_sample(
                sample_res=np.array([
                    res[arguments['indicator']]
                    for res in self.simulation_results
                ]),
                **options
            )

        if kind == "dynamic_ST":
            if arguments is None:
                raise ValueError("Please specify at least model output"
                                 " name as 'indicator")

            df_to_plot = pd.DataFrame({
                date: res['ST']
                for date, res in self.sensitivity_dynamic_results.items()
            }).T

            df_to_plot.columns = list(self.parameters_config.keys())

            options = {key: val for key, val in arguments.items()
                       if key != 'indicator'}

            plot_stacked_lines(df_to_plot, **options)


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


def plot_stacked_lines(
        df, title=None, y_label=None, x_label=None, legend_title=None):
    fig = go.Figure()
    for ind in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[ind],
            name=ind,
            mode='lines',
            stackgroup='one'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title=legend_title,
    )
    fig.show()


def plot_sample(sample_res, ref=None, title=None, y_label=None, x_label=None,
                x_axis=None, alpha=0.5):
    n_sample = sample_res.shape[0]
    x_to_plot = np.concatenate(
        [np.arange(sample_res.shape[1])] * n_sample)
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
                color=f'rgba(135, 135, 135, {alpha})',
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
