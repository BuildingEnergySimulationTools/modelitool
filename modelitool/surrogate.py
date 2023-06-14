import numpy as np
import pandas as pd

from modelitool.measure import time_series_control

from scipy.stats.qmc import LatinHypercube
from scipy.optimize import differential_evolution
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()


def get_aggregated_indicator(
    simulation_list,
    indicator,
    method=np.sum,
    method_args=None,
    reference=None,
    start=None,
    end=None,
):
    if not simulation_list:
        raise ValueError(
            "Empty simulation list. " "Cannot perform indicator aggregation"
        )

    if indicator not in list(simulation_list[0].columns):
        raise ValueError(
            "Indicator is not present in building_results or " "in energyplus_results"
        )

    y_df = pd.concat([sim[indicator] for sim in simulation_list], axis=1)

    if start is not None:
        if end is None:
            raise ValueError("If start is specified, " "end must also be specified")
        else:
            y_df = y_df.loc[start:end]
        if reference is not None:
            reference = reference.loc[start:end]

    if reference is None:
        return y_df.apply(method, axis=0).to_numpy()

    elif method_args is None:
        return np.array(
            [method(reference, y_df.iloc[:, i]) for i in range(y_df.shape[1])]
        )

    else:
        return np.array(
            [
                method(reference, y_df.iloc[:, i], **method_args)
                for i in range(y_df.shape[1])
            ]
        )


class SimulationSampler:
    def __init__(self, parameters, simulator, sampling_method="LatinHypercube"):
        self.parameters = parameters
        self.simulator = simulator
        self.sampling_method = sampling_method
        self.sample = np.empty(shape=(0, len(parameters)))
        self.sample_results = []
        if sampling_method == "LatinHypercube":
            self.sampling_method = LatinHypercube

    def get_boundary_sample(self):
        iter_index = list(itertools.product([0, 1], repeat=len(self.parameters)))

        return np.array(
            [
                [par["interval"][i] for par, i in zip(self.parameters, line)]
                for line in iter_index
            ]
        )

    def add_sample(self, sample_size, seed=None):
        sampler = LatinHypercube(d=len(self.parameters), seed=seed)
        new_sample = sampler.random(n=sample_size)
        new_sample_value = np.empty(shape=(0, len(self.parameters)))
        for s in new_sample:
            new_sample_value = np.vstack(
                (
                    new_sample_value,
                    [
                        par["interval"][0]
                        + val * par["interval"][1]
                        - par["interval"][0]
                        for par, val in zip(self.parameters, s)
                    ],
                )
            )

        # return new_sample_value

        if self.sample.size == 0:
            bound_sample = self.get_boundary_sample()
            new_sample_value = np.vstack((new_sample_value, bound_sample))

        prog_bar = progress_bar(range(new_sample_value.shape[0]))
        for pb, simul in zip(prog_bar, new_sample_value):
            sim_config = {
                param["name"]: val for param, val in zip(self.parameters, simul)
            }
            prog_bar.comment = "Simulations"
            self.simulator.set_param_dict(sim_config)
            self.simulator.simulate()
            results = self.simulator.get_results()
            self.sample_results.append(results)

        self.sample = np.vstack((self.sample, new_sample_value))


class SurrogateModel:
    def __init__(self, simulation_sampler):
        self.simulation_sampler = simulation_sampler
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.infos = {}

        self.model_dict = {
            "Tree_regressor": RandomForestRegressor(),
            "Random_forest": RandomForestRegressor(),
            "Linear_regression": LinearRegression(),
            "Linear_second_order": Pipeline(
                [("poly", PolynomialFeatures(2)), ("Line_reg", LinearRegression())]
            ),
            "Linear_third_order": Pipeline(
                [("poly", PolynomialFeatures(3)), ("Line_reg", LinearRegression())]
            ),
            "Support_Vector": SVR(),
            "Multi_layer_perceptron": MLPRegressor(max_iter=3000),
        }

    def add_samples(self, sample_size, seed=None):
        self.simulation_sampler.add_sample(sample_size, seed)

    def fit_sample(
        self,
        indicator="Total",
        start=None,
        end=None,
        metrics_method=mean_squared_error,
        aggregation_method=np.sum,
        method_args=None,
        reference=None,
        custom_series=None,
        verbose=True,
        random_state=None,
        test_size=0.2,
        cv=10,
    ):
        if custom_series is None:
            y_array = get_aggregated_indicator(
                simulation_list=self.simulation_sampler.sample_results,
                indicator=indicator,
                method=aggregation_method,
                method_args=method_args,
                reference=reference,
                start=start,
                end=end,
            )

        else:
            y_array = time_series_control(custom_series)

        x_scaled = self.x_scaler.fit_transform(self.simulation_sampler.sample)
        y_scaled = self.y_scaler.fit_transform(np.reshape(y_array, (-1, 1)))
        y_scaled = y_scaled.flatten()

        xs_train, xs_test, ys_train, ys_test = train_test_split(
            x_scaled, y_scaled, test_size=test_size, random_state=random_state
        )

        for key, mod in self.model_dict.items():
            mod.fit(xs_train, ys_train)

        score_dict = {}
        for key, mod in self.model_dict.items():
            cv_scores = cross_val_score(
                mod, xs_train, ys_train, scoring="neg_mean_squared_error", cv=cv
            )

            score_dict[key] = [np.mean(cv_scores), np.std(cv_scores)]
        sorted_score_dict = dict(
            sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        )

        if verbose:
            print(
                f"Cross validation neg_mean_squared_error scores"
                f"[mean, standard deviation] of {cv} folds"
            )
            print(sorted_score_dict)

        best_model_key = list(sorted_score_dict)[0]
        selected_mod = self.model_dict[best_model_key]
        ys_test_predicted = selected_mod.predict(xs_test)
        y_test = self.y_scaler.inverse_transform(np.reshape(ys_test, (-1, 1)))
        y_test_predicted = self.y_scaler.inverse_transform(
            np.reshape(ys_test_predicted, (-1, 1))
        )
        metrics_method_results = mean_squared_error(y_test, y_test_predicted)

        self.infos["best_model_key"] = best_model_key
        self.infos["metrics_method_results"] = metrics_method_results
        self.infos["metrics_method"] = metrics_method
        self.infos["indicator"] = indicator
        self.infos["aggregation_method"] = aggregation_method

        if verbose:
            print(f"{self.infos}")

    def predict(self, x_array):
        if self.infos == {}:
            raise ValueError(
                "Surrogate model is not fitted yet"
                "perform model fitting using fit_sample() method"
            )

        if x_array.ndim == 1:
            x_array = np.reshape(x_array, (1, -1))

        xs_array = self.x_scaler.transform(x_array)
        best_model = self.model_dict[self.infos["best_model_key"]]
        ys_array = best_model.predict(xs_array)
        return self.y_scaler.inverse_transform(np.reshape(ys_array, (-1, 1)))

    def minimization_identification(self):
        def objective_function(x):
            return self.predict(x)[0, 0]

        return differential_evolution(
            objective_function,
            bounds=[
                (param["interval"][0], param["interval"][1])
                for param in self.simulation_sampler.parameters
            ],
        )
