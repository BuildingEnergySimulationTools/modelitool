import os
import pandas as pd
from OMPython import ModelicaSystem
from OMPython import OMCSessionZMQ
from time import time


# TODO Create a debug mode to print time


class Simulator:
    def __init__(self,
                 model_path,
                 simulation_path,
                 simulation_options,
                 output_list,
                 init_parameters=None,
                 ):

        if not os.path.exists(simulation_path):
            os.mkdir(simulation_path)

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(
            f'cd("{simulation_path.as_posix()}")'
        )

        # A bit dirty but the only way I found to change the simulation dir
        # ModelicaSystem take cwd as currDirectory
        os.chdir(simulation_path)
        self.model = ModelicaSystem(
            fileName=model_path.as_posix(),
            modelName=model_path.stem,
        )

        self.output_list = output_list

        if init_parameters:
            self.set_param_dict(init_parameters)

        self.model.setSimulationOptions(
            [
                f'startTime={simulation_options["startTime"]}',
                f'stopTime={simulation_options["stopTime"]}',
                f'stepSize={simulation_options["stepSize"]}',
                f'tolerance={simulation_options["tolerance"]}',
                f'solver={simulation_options["solver"]}'
            ]
        )

    def set_param_dict(self, param_dict):
        # t1 = time()
        self.model.setParameters(
            [f"{item}={val}" for item, val in param_dict.items()]
        )
        # t2 = time()
        # print(f"Setting new parameters took {t2-t1}s")

    def simulate(self):
        # t1 = time()
        self.model.simulate()
        # t2 = time()
        # print(f"Simulating took {t2-t1}s")

    def get_results(self):
        # Modelica solver can provide several results for one timestep
        # Moreover variable timestep solver can provide messy result
        # Manipulations are done to resample the index and provide seconds
        # t1 = time()
        sol_list = self.model.getSolutions(["time"] + self.output_list).T

        res = pd.DataFrame(
            sol_list[:, 1:],
            index=sol_list[:, 0].flatten(),
            columns=self.output_list
        )

        res.columns = self.output_list

        res.index = pd.to_timedelta(res.index, unit='second')
        res = res.resample(
            f"{int(self.model.getSimulationOptions()['stepSize'])}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()
        res.index = res.index.astype('int')
        # t2 = time()
        # print(f"Getting results took {t2-t1}s")

        return res
