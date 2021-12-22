import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from modelitool.combitabconvert import datetime_to_seconds


class Identificator:
    def __init__(self,
                 simulator,
                 parameters,
                 error_function=None
                 ):
        self.simulator = simulator
        self.param_init = {
            item: parameters[item]["init"]
            for item, val in parameters.items()
        }
        self.param_interval = {
            item: parameters[item]["interval"]
            for item, val in parameters.items()
        }

        self.param_identified = {
            item: np.nan for item in parameters.keys()
        }

        self.simulator.set_param_dict(self.param_init)

        if error_function is None:
            self.error_function = mean_squared_error
        else:
            self.error_function = error_function

    def fit(self, features, labels):
        print('Optimization started')
        print(self.param_interval)

        if features is not None:
            self.simulator.set_boundaries_df(features)
            dymo_index = datetime_to_seconds(features.index)
            self.simulator.model.setSimulationOptions(
                [
                    f'startTime={dymo_index[0]}',
                    f'stopTime={dymo_index[-1]}',
                ]
            )

        res = differential_evolution(
            self._objective_function,
            args=(labels, ),
            bounds=list(self.param_interval.values()),
            popsize=5,
            callback=self._optimization_callback
        )

        if res['success']:
            print(f'Identification successful error function = {res["fun"]}')
            for key, val in zip(self.param_identified.keys(), res["x"]):
                self.param_identified[key] = val
        else:
            raise ValueError("Identification failed to converge")

    def predict(self, features):
        if list(self.param_identified.values()) == \
                [np.nan] * len(self.param_identified):
            raise ValueError(
                "Parameters have not been identified, please fit model"
            )
        else:
            self.simulator.set_param_dict(self.param_identified)
            self.simulator.set_boundaries_df(features)
            dymo_index = datetime_to_seconds(features.index)
            self.simulator.model.setSimulationOptions(
                [
                    f'startTime={dymo_index[0]}',
                    f'stopTime={dymo_index[-1]}',
                ]
            )
            self.simulator.simulate()
            return self.simulator.get_results(
                index_datetime=True,
                ref_year=features.index[0].year
            )

    def _optimization_callback(self, xk, convergence):
        print({
            it: val for it, val in zip(self.param_init.keys(), xk)
        })
        print(f'convergence = {convergence}')

    def _objective_function(self, x, *args):
        labels, = args
        tempo_dict = {
            item: x[i] for i, item in enumerate(self.param_init.keys())
        }
        self.simulator.set_param_dict(tempo_dict)
        self.simulator.simulate()

        return self.error_function(
            self.simulator.get_results(), labels
        )