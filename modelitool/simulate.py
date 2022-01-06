import os
import pandas as pd
import datetime as dt
import time

import numpy as np


from OMPython import ModelicaSystem
from OMPython import OMCSessionZMQ
import tempfile
from pathlib import Path

from modelitool.combitabconvert import df_to_combitimetable
from modelitool.combitabconvert import seconds_to_datetime


# TODO Create a debug mode to print time


class Simulator:
    def __init__(self,
                 model_path,
                 simulation_options,
                 output_list,
                 init_parameters=None,
                 simulation_path=None,
                 boundary_df=None,
                 package_path=None,
                 lmodel=[]
                 ):
        if type(model_path) == str:
            model_path = Path(model_path)

        if simulation_path is None:
            simulation_path = tempfile.mkdtemp()
            simulation_path = Path(simulation_path)

        if not os.path.exists(simulation_path):
            os.mkdir(simulation_path)

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(
            f'cd("{simulation_path.as_posix()}")'
        )

        # A bit dirty but the only way I found to change the simulation dir
        # ModelicaSystem take cwd as currDirectory
        os.chdir(simulation_path)
        if package_path is None:
            model_system_args = {
                'fileName': model_path.as_posix(),
                'modelName': model_path.stem,
                'lmodel': lmodel
            }
        else:
            model_system_args = {
                'fileName': package_path.as_posix(),
                'modelName': model_path,
                'lmodel': lmodel
            }

        self.model = ModelicaSystem(**model_system_args)

        self._simulation_path = simulation_path
        self.output_list = output_list

        if boundary_df is not None:
            self.set_boundaries_df(boundary_df)
            self.boundaries = boundary_df
        else:
            self.boundaries = None

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

    def get_available_outputs(self):
        if self.model.getSolutions() is None:
            # A bit dirty but simulation must be run once so
            # getSolutions() can access results
            self.simulate()

        return self.model.getSolutions()

    def set_boundaries_df(self, df):
        # DataFrame columns order must match the order
        # defined in the modelica file. This cannot be checked
        # Modelica file must contain a combiTimetable named Boundaries

        new_bounds_path = self._simulation_path / "bounds.txt"
        df_to_combitimetable(df, new_bounds_path)
        self.model.setParameters(
            f'Boundaries.fileName="{new_bounds_path.as_posix()}"'
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
        self.model.simulate(resultfile='res.mat')
        # t2 = time()
        # print(f"Simulating took {t2-t1}s")

    def get_results(self, index_datetime=False, ref_year=2009):
        # Modelica solver can provide several results for one timestep
        # Moreover variable timestep solver can provide messy result
        # Manipulations are done to resample the index and provide seconds
        # t1 = time()
        sol_list = self.model.getSolutions(
            ["time"] + self.output_list,
            resultfile='res.mat'
        ).T

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

        if not index_datetime:
            res.index = res.index.astype('int')
        else:
            res.index = seconds_to_datetime(res.index, ref_year)
        # t2 = time()
        # print(f"Getting results took {t2-t1}s")
        return res


def run_batch(simulator, param_name_list, sample, verbose_step=10):
    simu_list = []
    for samp in sample:
        simu_list.append({
            param: val for param, val in zip(
                param_name_list, samp
            )
        })

    # A bit dirty but perform first simulation to get
    # output shape and simulation time estimation
    t1 = time.time()

    simulator.set_param_dict(simu_list[0])
    simulator.simulate()
    results = simulator.get_results()

    t2 = time.time()
    sim_duration = dt.timedelta(seconds=t2 - t1)

    simulation_results = np.zeros((
        sample.shape[0],
        results.shape[0],
        results.shape[1]
    ))

    simulation_results[0] = results.to_numpy()

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

        simulator.set_param_dict(sim)
        simulator.simulate()
        results = simulator.get_results()
        simulation_results[idx + 1] = results.to_numpy()

    return simulation_results
