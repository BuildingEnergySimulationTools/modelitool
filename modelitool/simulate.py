import os
import warnings

import pandas as pd
import datetime as dt

from pydelica import Session
from pydelica.logger import OMLogLevel
from pydelica.options import Solver

import tempfile
from pathlib import Path

from modelitool.combitabconvert import df_to_combitimetable
from modelitool.combitabconvert import seconds_to_datetime


class Simulator:
    def __init__(
        self,
        model_path,
        model_name=None,
        simulation_options=None,
        output_list=None,
        init_parameters=None,
        boundary_df=None,
        year=None,

    ):
        if isinstance(model_path, str):
            self.model_path = Path(model_path)
        else:
            self.model_path = model_path

        if boundary_df is not None:
            self.set_boundaries_df(boundary_df)
            if year is not None:
                warnings.warn(
                    "Simulator year is read from boundary"
                    "DAtaFrame. Argument year is ignored"
                )
        elif year is not None:
            self.year = year
        else:
            self.year = dt.date.today().year

        if init_parameters:
            self.set_param_dict(init_parameters)

        self.simulation_options = simulation_options

        self.session = Session(log_level=OMLogLevel.STATS)
        self.session.__enter__()
        self.build_model()
        if model_name is None:
            self.model_name = list(self.session._model_parameters.keys())[0]

        if simulation_options is not None:
            self.set_simulation_options(simulation_options)

        if output_list is not None:
            self.set_variable_filter(output_list, self.model_name)
            self.output_list = output_list
    def __del__(self):
        self.session.__exit__()

    def build_model(self, profiling=None, omc_build_flags=None):
        self.session.build_model(self.model_path, profiling, omc_build_flags)

    def get_parameters(self):
        return self.session.get_parameters()

    def set_variable_filter(self, output_list, model_name=None):
        if model_name is None:
            model_name = self.model_name

        self.session.set_variable_filter("|".join(output_list), model_name=model_name)

    def set_simulation_options(self, simulation_options):

        self.session.set_solver(getattr(Solver, simulation_options["solver"].upper()))
        self.session.set_time_range(
            start_time=simulation_options["startTime"],
            stop_time=simulation_options["stopTime"]
        )
        self.session.set_tolerance(tolerance=simulation_options["tolerance"])

        for model in self.session._simulation_opts:
            self.session._simulation_opts[model].set_option(
                "stepSize", simulation_options["stepSize"])

        self.simulation_options = simulation_options

    def set_boundaries_df(self, df):
        # DataFrame columns order must match the order
        # defined in the modelica file. This cannot be checked
        # Modelica file must contain a combiTimetable named Boundaries

        new_bounds_path = self._simulation_path / "bounds.txt"
        df_to_combitimetable(df, new_bounds_path)
        self.model.setParameters(f'Boundaries.fileName="{new_bounds_path.as_posix()}"')
        try:
            self.year = df.index[0].year
        except ValueError:
            raise ValueError(
                "Could not read date from boundary condition. "
                "Please verify that Dataframe index is a datetime"
            )

    def set_param_dict(self, param_dict):
        for key, value in param_dict.items():
            self.session.set_parameter(key, value)

    def simulate(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.session.simulate(model_name=model_name)

    def get_results(self, model_name=None, index_datetime=True):
        # Modelica solver can provide several results for one timestep
        # Moreover variable timestep solver can provide messy result
        # Manipulations are done to resample the index and provide seconds
        if model_name is None:
            model_name = self.model_name

        res = pd.DataFrame(self.session.get_solutions()[model_name])
        res.index = res['time']
        res = res.drop('time', axis=1)

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{self.session._simulation_opts['rosen'].get('stepSize')}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()

        if not index_datetime:
            res.index = res.index.astype("int")
        else:
            res.index = seconds_to_datetime(res.index, self.year)

        return res
