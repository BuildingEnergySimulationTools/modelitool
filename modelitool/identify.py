import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from modelitool.combitabconvert import datetime_to_seconds
import time


class Identificator:
    def __init__(self, simulator, parameters, error_function=None):
        self.simulator = simulator
        self.param_init = {param["name"]: param["init"] for param in parameters}
        self.param_interval = {param["name"]: param["interval"] for param in parameters}

        self.param_identified = {param["name"]: np.nan for param in parameters}

        self.simulator.set_param_dict(self.param_init)

        if error_function is None:
            self.error_function = mean_squared_error
        else:
            self.error_function = error_function

        self.optimization_results = None

    def fit(
        self,
        features,
        labels,
        convergence_tolerance=0.05,
        population_size=15,
        crossover_probability=0.7,
        mutation_constant=(0.5, 1),
        max_iteration=1000,
    ):
        print("Optimization started")
        print(self.param_interval)
        start_time_eval = time.time()

        if features is not None:
            self.simulator.set_combi_time_table_df(
                features, combi_time_table_name="Boundaries"
            )
            dymo_index = datetime_to_seconds(features.index)
            self.simulator.set_simulation_options(
                {
                    "startTime": dymo_index[0],
                    "stopTime": dymo_index[-1],
                }
            )

        res = differential_evolution(
            self._objective_function,
            args=(labels,),
            bounds=list(self.param_interval.values()),
            callback=self._optimization_callback,
            popsize=population_size,
            tol=convergence_tolerance,
            recombination=crossover_probability,
            mutation=mutation_constant,
            maxiter=max_iteration,
        )

        if res["success"]:
            print(f'Identification successful error function = {res["fun"]}')
            for key, val in zip(self.param_identified.keys(), res["x"]):
                self.param_identified[key] = val
            self.optimization_results = res
        else:
            raise ValueError("Identification failed to converge")

        print("Duration: {}".format(time.time() - start_time_eval))

    def predict(self, features):
        if list(self.param_identified.values()) == [np.nan] * len(
            self.param_identified
        ):
            raise ValueError("Parameters have not been identified, please fit model")
        else:
            self.simulator.set_param_dict(self.param_identified)
            self.simulator.set_combi_time_table_df(
                features, combi_time_table_name="Boundaries"
            )
            dymo_index = datetime_to_seconds(features.index)
            self.simulator.set_simulation_options(
                {
                    "startTime": dymo_index[0],
                    "stopTime": dymo_index[-1],
                }
            )
            self.simulator.simulate()
            return self.simulator.get_results()

    def _optimization_callback(self, xk, convergence):
        print({it: val for it, val in zip(self.param_init.keys(), xk)})
        print(f"convergence = {convergence}")

    def _objective_function(self, x, *args):
        (labels,) = args
        tempo_dict = {item: x[i] for i, item in enumerate(self.param_init.keys())}
        self.simulator.set_param_dict(tempo_dict)
        self.simulator.simulate()

        return self.error_function(self.simulator.get_results(), labels)
