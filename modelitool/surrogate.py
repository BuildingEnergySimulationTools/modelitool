import numpy as np

from copy import deepcopy
from scipy.stats.qmc import LatinHypercube
import itertools

from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()


class SimulationSampler:
    def __init__(
            self,
            parameters,
            simulator,
            sampling_method='LatinHypercube'):

        self.parameters = parameters
        self.simulator = simulator
        self.sampling_method = sampling_method
        self.sample = np.empty(shape=(0, len(parameters)))
        self.sample_results = []
        if sampling_method == 'LatinHypercube':
            self.sampling_method = LatinHypercube

    def get_boundary_sample(self):
        iter_index = list(
            itertools.product([0, 1], repeat=len(self.parameters)))

        return np.array([
            [self.parameters[par][i] for par, i in zip(self.parameters, line)]
            for line in iter_index
        ])

    def add_sample(self,
                   sample_size,
                   seed=None):

        sampler = LatinHypercube(d=len(self.parameters), seed=seed)
        new_sample = sampler.random(n=sample_size)
        new_sample_value = np.empty(shape=(0, len(self.parameters)))
        for s in new_sample:
            new_sample_value = np.vstack((
                new_sample_value,
                [
                    (self.parameters[par][0] +
                     val * (self.parameters[par][1] - self.parameters[par][0]))
                    for par, val in zip(self.parameters, s)
                ]
            ))

        if self.sample.size == 0:
            bound_sample = self.get_boundary_sample()
            new_sample_value = np.vstack((new_sample_value, bound_sample))

        prog_bar = progress_bar(range(new_sample_value.shape[0]))
        for pb, simul in zip(prog_bar, new_sample_value):
            sim_config = {
                param: val for param, val in zip(self.parameters.keys(), simul)
            }
            prog_bar.comment = 'Simulations'
            self.simulator.set_param_dict(sim_config)
            self.simulator.simulate()
            results = self.simulator.get_results()
            self.sample_results.append(results)

        self.sample = np.vstack((self.sample, new_sample_value))
