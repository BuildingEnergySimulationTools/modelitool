from SALib.sample import fast_sampler
from SALib.sample import saltelli
from SALib.sample import morris as morris_sampler
from SALib.sample import latin

from SALib.analyze import fast
from SALib.analyze import morris
from SALib.analyze import sobol
from SALib.analyze import rbd_fast

from fastprogress.fastprogress import force_console_behavior

import plotly.graph_objects as go

import numpy as np
import pandas as pd

master_bar, progress_bar = force_console_behavior()


def check_arguments(res, param, result):
    if result not in res:
        raise ValueError(f"{result} not found in results")

    if res[result].shape[0] != len(param):
        raise ValueError("Parameters name and index length mismatch")


def modelitool_to_salib_problem(modelitool_problem):
    return {
        "num_vars": len(modelitool_problem),
        "names": [p["name"] for p in modelitool_problem],
        "bounds": list(map(lambda p: p["interval"], modelitool_problem)),
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
            "Morris": {"method": morris, "sampling": morris_sampler},
            "Sobol": {
                "method": sobol,
                "sampling": saltelli,
            },
            "RBD_fast": {
                "method": rbd_fast,
                "sampling": latin,
            },
        }
        self.simulator = simulator
        self.parameters_config = parameters_config
        if sensitivity_method not in self.meth_samp_map.keys():
            raise ValueError("Specified sensitivity method is not valid")
        else:
            self._sensitivity_method = sensitivity_method

        self.salib_problem = modelitool_to_salib_problem(self.parameters_config)

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

        sampler = self.meth_samp_map[self._sensitivity_method]["sampling"]
        sample_temp = sampler.sample(N=n, problem=self.salib_problem, **arguments)

        for index, param in enumerate(self.parameters_config):
            vtype = param["type"]
            if vtype == "Integer":
                sample_temp[:, index] = np.round(sample_temp[:, index])
                sample_temp = np.unique(sample_temp, axis=0)
        self.sample = sample_temp

    def run_simulations(self, verbose_step=10, simflags=None):
        if self.sample.size == 0:
            raise ValueError("No sample available. Generate sample using draw_sample()")

        self.simulation_results = list()
        simu_list = []

        for samp in self.sample:
            simu_list.append(
                {param["name"]: val for param, val in zip(self.parameters_config, samp)}
            )

        prog_bar = progress_bar(range(len(simu_list)))

        for pb, sim in zip(prog_bar, simu_list):
            prog_bar.comment = "Simulations"
            self.simulator.set_param_dict(sim)
            self.simulator.simulate(simflags=simflags)
            results = self.simulator.get_results()
            self.simulation_results.append(results)

    def _compute_aggregated(self, aggregation_method, indicator, ref=None, freq=None):
        aggregated_list = []

        if freq is None:
            for res in self.simulation_results:
                if ref is None:
                    aggregated_list.append(aggregation_method(res[indicator]))
                else:
                    aggregated_list.append(aggregation_method(res[indicator], ref))
        else:
            grouper = pd.Grouper(freq=freq)
            prog_bar = progress_bar(range(len(self.simulation_results)))

            for mb, res in zip(prog_bar, self.simulation_results):
                tempos = pd.Series()
                simu_group = res[indicator].groupby(grouper)
                prog_bar.comment = "Aggregation"

                if ref is not None:
                    ref_group = ref.groupby(grouper)
                    for simu_gr, ref_gr in zip(simu_group, ref_group):
                        tempos[simu_gr[0]] = aggregation_method(ref_gr[1], simu_gr[1])
                else:
                    for simu_gr in simu_group:
                        tempos[simu_gr[0]] = aggregation_method(simu_gr[1])

                aggregated_list.append(tempos)

        return aggregated_list

    def analyze(self, indicator, aggregation_method, reference=None, arguments=None):
        if arguments is None:
            arguments = {}

        if indicator not in self.simulator_outputs:
            raise ValueError("Specified indicator not in computed outputs")

        analyser = self.meth_samp_map[self._sensitivity_method]["method"]

        y_array = np.array(
            self._compute_aggregated(aggregation_method, indicator, reference)
        )

        if self._sensitivity_method in ["Sobol", "FAST"]:
            self.sensitivity_results = analyser.analyze(
                problem=self.salib_problem, Y=y_array, **arguments
            )
        elif self._sensitivity_method in ["Morris", "RBD_fast"]:
            self.sensitivity_results = analyser.analyze(
                problem=self.salib_problem, X=self.sample, Y=y_array, **arguments
            )

    def dynanalyze(
        self,
        indicator,
        aggregation_method,
        reference=None,
        freq=None,
        absolute=False,
        arguments=None,
    ):
        if arguments is None:
            arguments = {}

        if freq is None:
            raise ValueError("Specify a frequency for dynamic analysis")

        analyser = self.meth_samp_map[self._sensitivity_method]["method"]

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
            prog_bar.comment = "Dynamic index"

            if self._sensitivity_method in ["Sobol", "FAST"]:
                self.sensitivity_dynamic_results[idx] = analyser.analyze(
                    problem=self.salib_problem, Y=res, **arguments
                )

            elif self._sensitivity_method in ["Morris", "RBD_fast"]:
                self.sensitivity_dynamic_results[idx] = analyser.analyze(
                    problem=self.salib_problem, X=self.sample, Y=res, **arguments
                )

        if absolute:
            numpy_var = np.var(numpy_res, axis=1)

            if self._sensitivity_method in ["Sobol", "FAST"]:
                for key, dict_res, var in zip(
                    self.sensitivity_dynamic_results.keys(),
                    self.sensitivity_dynamic_results.values(),
                    numpy_var,
                ):
                    for idx in dict_res.keys():
                        self.sensitivity_dynamic_results[key][idx] *= var

            else:
                for idx, it in self.sensitivity_dynamic_results.items():
                    it.pop("names", None)

                res_array = {
                    idx: {idx_n: np.array(it_n) for idx_n, it_n in it.items()}
                    for idx, it in self.sensitivity_dynamic_results.items()
                }
                for key, dict_res, var in zip(
                    res_array.keys(), res_array.values(), numpy_var
                ):
                    for idx in dict_res.keys():
                        res_array[key][idx] *= var
                self.sensitivity_dynamic_results = res_array

    def plot(self, kind="bar", arguments=None):
        if kind == "bar":
            if self.sensitivity_results is None:
                raise ValueError("No result to plot")

            if self._sensitivity_method == "Sobol":
                plot_sobol_st_bar(
                    self.sensitivity_results,
                    [param["name"] for param in self.parameters_config],
                )

        if kind == "parallel":
            if arguments is None:
                raise ValueError(
                    "Please provide list of dict to compute " "indicators values"
                )

            ind_dict = {}
            for ind in arguments["indicator_dict_list"]:
                arg_dict = {
                    key: ind[key]
                    for key in ind.keys()
                    if key in ["aggregation_method", "indicator", "ref"]
                }

                ind_dict[ind["name"]] = np.array(self._compute_aggregated(**arg_dict))

            param_dict = {
                par["name"]: values
                for par, values in zip(self.parameters_config, self.sample.T)
            }

            if "plot_options" not in arguments.keys():
                options = {"colorby": list(ind_dict.keys())[0]}
            else:
                options = arguments["plot_options"]

            plot_parcoord(data_dict={**param_dict, **ind_dict}, **options)

        if kind == "sample":
            if arguments is None:
                raise ValueError(
                    "Please specify at least model output" " name as 'indicator"
                )

            options = {key: val for key, val in arguments.items() if key != "indicator"}

            plot_sample(
                sample_res=[
                    res[arguments["indicator"]] for res in self.simulation_results
                ],
                **options,
            )

        if kind == "dynamic_ST":
            if arguments is None:
                raise ValueError(
                    "Please specify at least model output" " name as 'indicator"
                )

            if self._sensitivity_method == "RBD_fast":
                indic = "S1"
            elif self._sensitivity_method == "Sobol":
                indic = "ST"
            else:
                raise ValueError("Invalid sensitivity method")

            df_to_plot = pd.DataFrame(
                {
                    date: res[indic]
                    for date, res in self.sensitivity_dynamic_results.items()
                }
            ).T

            df_to_plot.columns = [p["name"] for p in self.parameters_config]

            options = {key: val for key, val in arguments.items() if key != "indicator"}

            plot_stacked_lines(df_to_plot, **options)


def plot_parcoord(data_dict, colorby=None, colorscale="Electric"):
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=data_dict[colorby],
                colorscale=colorscale,
                showscale=True,
                cmin=data_dict[colorby].min(),
                cmax=data_dict[colorby].max(),
            ),
            dimensions=[
                {
                    "range": [data_dict[par].min(), data_dict[par].max()],
                    "label": par,
                    "values": data_dict[par],
                }
                for par in data_dict.keys()
            ],
        )
    )

    fig.show()


def plot_sobol_st_bar(salib_res, param_names):
    check_arguments(salib_res, param_names, "ST")

    sobol_ind = salib_res.to_df()[0]
    sobol_ind.sort_values(by="ST", ascending=True, inplace=True)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=sobol_ind.index,
            y=sobol_ind.ST,
            name="Sobol Total Indices",
            marker_color="orange",
            error_y=dict(type="data", array=sobol_ind.ST_conf.to_numpy()),
            yaxis="y1",
        )
    )

    figure.update_layout(
        title="Sobol Total indices",
        xaxis_title="Parameters",
        yaxis_title="Sobol total index value [0-1]",
    )

    figure.show()


def plot_stacked_lines(df, title=None, y_label=None, x_label=None, legend_title=None):
    fig = go.Figure()
    for ind in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[ind], name=ind, mode="lines", stackgroup="one")
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title=legend_title,
    )
    fig.show()


def plot_sample(
    sample_res,
    ref=None,
    title=None,
    y_label=None,
    x_label=None,
    alpha=0.5,
    loc=None,
):
    if ref is not None:
        try:
            to_plot = pd.concat([res.loc[ref.index] for res in sample_res])
        except ValueError:
            raise ValueError("Provide Pandas Series or DataFrame as ref")

    else:
        to_plot = pd.concat([res for res in sample_res])

    if loc is not None:
        to_plot = pd.concat([res for res in sample_res]).loc[loc[0]:loc[1]]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            name="Sample",
            mode="markers",
            x=to_plot.index,
            y=np.array(to_plot),
            marker=dict(
                color=f"rgba(135, 135, 135, {alpha})",
            ),
        )
    )

    if ref is not None:
        fig.add_trace(
            go.Scattergl(
                name="Reference",
                mode="lines",
                x=ref.index,
                y=ref,
                line=dict(color="crimson", width=2),
            )
        )

    if title is not None:
        fig.update_layout(title=title)
    if y_label is not None:
        fig.update_layout(yaxis_title=y_label)

    if x_label is not None:
        fig.update_layout(xaxis_title=x_label)

    fig.show()


def plot_morris_scatter(salib_res, title=None, unit="", scaler=100, autosize=True):
    morris_res = salib_res.to_df()
    morris_res["distance"] = np.sqrt(morris_res.mu_star**2 + morris_res.sigma**2)
    morris_res["dimless_distance"] = morris_res.distance / morris_res.distance.max()

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=morris_res.mu_star,
            y=morris_res.sigma,
            name="Morris index",
            mode="markers+text",
            text=list(morris_res.index),
            textposition="top center",
            marker=dict(
                size=morris_res.dimless_distance * scaler,
                color=np.arange(morris_res.shape[0]),
            ),
            error_x=dict(
                type="data",  # value of error bar given in data coordinates
                array=morris_res.mu_star_conf,
                color="#696969",
                visible=True,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 0.1 * morris_res.mu_star.max() * 1.1]),
            name="linear_lim",
            mode="lines",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 0.5 * morris_res.mu_star.max() * 1.1]),
            name="Monotonic limit",
            mode="lines",
            line=dict(color="grey", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([0, morris_res.mu_star.max() * 1.1]),
            y=np.array([0, 1 * morris_res.mu_star.max() * 1.1]),
            name="Non linear limit",
            mode="lines",
            line=dict(color="grey", dash="dashdot"),
        )
    )

    # Edit the layout
    if title is not None:
        title = title
    else:
        title = "Morris Sensitivity Analysis"

    if autosize:
        y_lim = [-morris_res.sigma.max() * 0.1, morris_res.sigma.max() * 1.5]
    else:
        y_lim = [-morris_res.sigma.max() * 0.1, morris_res.mu_star.max() * 1.1]

    x_label = f"Absolute mean of elementary effects μ* [{unit}]"
    y_label = f"Standard deviation of elementary effects σ [{unit}]"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_lim,
    )

    fig.show()
