from modelitool.simulate import run_batch
from modelitool.sensitivity import modelitool_to_salib_problem

import pandas as pd

import numpy as np
from SALib.sample import latin

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pymc3 as pm

import arviz as az

import theano.tensor as tt
import theano

from scipy.stats import truncnorm


class TruncatedNormal:
    """Mimic PYMC TruncatedNormal"""
    def __init__(self,
                 name,
                 mu,
                 sigma,
                 lower,
                 upper):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

        # See scipy doc
        self._a, self._b = (0 - mu) / sigma, (1 - mu) / sigma

    def sample(self, size=1000):
        return truncnorm.rvs(
            self._a, self._b, loc=self.mu, scale=self.sigma, size=size)


class DCGenerator:
    def __init__(self,
                 simulator,
                 params,
                 sample_size,
                 observable_inputs=None,
                 random_seed=42):

        self.simulator = simulator
        self.params = params
        self.sample_size = sample_size
        self.sample = self.draw_sample(seed=random_seed)
        self.simulation_results = np.array([])
        self.eta_scaler = StandardScaler()
        self.x_scaler = MinMaxScaler()
        self.t_scaler = MinMaxScaler()

        if isinstance(observable_inputs, str) and \
                observable_inputs == "from_simulator":
            if simulator.boundaries is not None:
                self.observable_inputs = simulator.boundaries
            else:
                raise ValueError("Simulator doesn't have a "
                                 "boundary df")

        elif observable_inputs is not None:
            self.observable_inputs = observable_inputs

        else:
            warnings.warn("No observable input were specified")

    def get_dc(self, indicator, scaled=True):
        if self.simulation_results.size == 0:
            raise ValueError("Empty simulation results")

        if self.observable_inputs is not None:
            eta = self.simulation_results[
                      ...,
                      self.simulator.output_list.index(indicator)
                  ].flatten()[:, np.newaxis]

            if isinstance(self.observable_inputs, pd.DataFrame):
                np_observable = self.observable_inputs.to_numpy()
            elif isinstance(self.observable_inputs, np.array):
                np_observable = self.observable_inputs
            else:
                raise ValueError("Please provide pd.DataFrame Object"
                                 "for observable input")
            observable = np.concatenate(
                [np_observable] * self.sample_size)

            parameters = np.concatenate([
                np.concatenate(
                    [sim[:, np.newaxis].T] * self.simulation_results.shape[1])
                for sim in self.sample
            ])

            self.eta_scaler.fit(eta)
            self.x_scaler.fit(observable)
            self.t_scaler.fit(parameters)

            if scaled:
                scaled_eta = self.eta_scaler.transform(eta)
                scaled_obs = self.x_scaler.transform(observable)
                scaled_params = self.t_scaler.transform(parameters)
                return np.concatenate([
                    scaled_eta, scaled_obs, scaled_params], axis=1)
            else:
                return np.concatenate([
                    eta, observable, parameters], axis=1)

    def draw_sample(self, seed=None):
        salib_problem = modelitool_to_salib_problem(self.params)
        return latin.sample(salib_problem, N=self.sample_size, seed=seed)

    def run_simulations(self, verbose_step=10):
        self.simulation_results = run_batch(
            simulator=self.simulator,
            param_name_list=list(self.params.keys()),
            sample=self.sample,
            verbose_step=verbose_step
        )

    def plot_param_distribution(self):

        for i, par in enumerate(self.params.keys()):
            sns.displot(self.sample[:, i])

        plt.show()


class PYMCGPTrainer:
    def __init__(self,
                 x_observed,
                 y_observation,
                 kernel=None):
        self.x_observed = x_observed
        self.y_observation = y_observation
        if kernel is None:
            self.kernel = pm.gp.cov.ExpQuad
        self.map_values = None
        self.fitting_trace = None
        self.gp_fitting = None
        self.gp_regression = None

    def fit(self, id_method="MAP", trace_length=1000, tuning=500,
            chains=2, cores=1):
        """
        Other choice for id_method is "trace",
        If trace is used, user shall also specify "trace_length"
        "tuning", number of chains to sample "chains", and number of
        CPU to use "core". Note that "core" is set to 1 by default.
        Trouble may arise when multiprocessing on Windows machine
        """

        self.gp_fitting = pm.Model()

        with self.gp_fitting:
            sigma_model = pm.HalfNormal('sigma_model', 0.1)
            rho = pm.Lognormal('rho', mu=0, sigma=3)

            k = self.kernel(input_dim=self.x_observed.shape[1], ls=rho)
            model = pm.gp.Marginal(cov_func=k)

            model.marginal_likelihood(
                'mod_out',
                X=self.x_observed,
                y=self.y_observation,
                noise=sigma_model
            )

        if id_method == "MAP":
            with self.gp_fitting:
                map_estimate, optim_result = pm.find_MAP(
                    progressbar=True,
                    method="L-BFGS-B",
                    return_raw=True,
                )

            self.map_values = map_estimate

        elif id_method == "trace":
            with self.gp_fitting:
                self.fitting_trace = pm.sample(
                    trace_length,
                    chains=chains,
                    tune=tuning,
                    cores=cores,
                    return_inferencedata=True
                )

    def plot_fitting_trace(self):
        if self.fitting_trace is not None:
            with self.gp_fitting:
                az.plot_trace(self.fitting_trace);
        else:
            raise ValueError("No fitting trace were found."
                             "Use 'fit' method with id_method='trace' ")

    def predict(self,
                x_prediction,
                method="MAP",
                samples=100,
                trace_length=100,
                tuning=50,
                chains=2,
                cores=1):

        self.gp_regression = pm.Model()

        if method == "trace":
            # TODO implement trace method
            raise ValueError("Not yet implemented")

        if method == "MAP":
            if self.map_values is None:
                raise ValueError("No MAP values were found for the GP"
                                 " hyperparameters. Use 'fit' method with "
                                 "id_method='MAP'")
            else:
                with self.gp_regression:
                    # Things here are very questionable.
                    # Model noise "sigma_model" is 0 centered with very
                    # low 'sigma'.
                    # MAP estimate for sigma probably very close 0
                    # But later sampling doesn't like it when it's < 10E-5
                    # So if map["sigma_model"] < 10E-6
                    # We juste set it to 10E-6

                    if self.map_values["sigma_model"] < 10E-6:
                        sigma_model = pm.HalfNormal('sigma_model', 10E-6)
                    else:
                        sigma_model = pm.HalfNormal(
                            'sigma_model', self.map_values["sigma_model"])

                    k = pm.gp.cov.ExpQuad(
                        input_dim=x_prediction.shape[1],
                        ls=self.map_values["rho"])

                    reg_model = pm.gp.Marginal(cov_func=k)

                    reg_model.marginal_likelihood(
                        'Tins',
                        X=self.x_observed,
                        y=self.y_observation,
                        noise=sigma_model)

        with self.gp_regression:
            reg_out = reg_model.conditional('reg_out', x_prediction)

        with self.gp_regression:
            return pm.sample_posterior_predictive(
                [self.map_values],
                var_names=["reg_out"],
                samples=samples,
            )
