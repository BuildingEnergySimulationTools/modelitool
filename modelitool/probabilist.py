from modelitool.simulate import run_batch

import pandas as pd

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


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

    def get_dc(self, indicator):
        if self.simulation_results.size == 0:
            raise ValueError("Empty simulation results")

        if self.observable_inputs is not None:
            simulated_outputs = self.simulation_results[
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

            return np.concatenate([
                simulated_outputs,
                observable,
                parameters
            ], axis=1)

    def draw_sample(self, seed=None):
        gather_list = []
        for val in self.params.values():
            if seed is not None:
                rs = RandomState(MT19937(SeedSequence(seed)))
                uniform_pdf = rs.uniform
            else:
                uniform_pdf = np.random.uniform

            gather_list.append(uniform_pdf(low=val[0], high=val[1],
                                           size=self.sample_size))

        return np.concatenate([gather_list], axis=1).T

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
