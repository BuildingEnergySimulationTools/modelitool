from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution


class Identificator:
    def __init__(self,
                 simulator,
                 parameters,
                 y_train,
                 x_train=None):

        # First dirty version x_train is not used
        self.simulator = simulator
        self.param_init = {
            item: parameters[item]["init"]
            for item, val in parameters.items()
        }
        self.param_interval = {
            item: parameters[item]["interval"]
            for item, val in parameters.items()
        }
        self.x_train = x_train
        self.y_train = y_train

        self.simulator.set_param_dict(self.param_init)

    def _objective_function(self, x):
        tempo_dict = {
            item: x[i] for i, item in enumerate(self.param_init.keys())
        }
        self.simulator.set_param_dict(tempo_dict)
        self.simulator.simulate()

        return mean_squared_error(
            self.simulator.get_results(), self.y_train
        )

    def fit(self):
        print('Optimization started')
        print(self.param_interval)
        res = differential_evolution(
            self._objective_function,
            bounds=list(self.param_interval.values()),
            popsize=5,
            callback=self._optimization_callback
        )
        return res

    def _optimization_callback(self, xk, convergence):
        print({
            it: val for it, val in zip(xk, self.param_init.keys())
        })
        print(f'convergence = {convergence}')
