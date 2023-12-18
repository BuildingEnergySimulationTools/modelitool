import warnings

import pandas as pd
import datetime as dt

from pydelica import Session
from pydelica.logger import OMLogLevel
from pydelica.options import Solver

from pathlib import Path

from modelitool.combitabconvert import df_to_combitimetable, seconds_to_datetime

from corrai.base.model import Model

from datetime import datetime

from copy import deepcopy


def seconds_since_start_of_year(date_str: str):
    """
    Calculate the number of seconds since the beginning of the year from a given date and time.

    Parameters:
    - date_str (str): A string representing the date and time in the format 'yyyy-mm-dd hh:mm:ss'.

    Returns:
    - int: The number of seconds since the start of the year for the given date and time.

    If the input date_str is not in the correct format, it raises a ValueError with the message
    'Invalid date format. Please use 'yyyy-mm-dd hh:mm:ss'.'

    Example:
    >>> date_str = "2023-09-19 15:30:00"
    >>> seconds_since_start_of_year(date_str)
    6078000  # This is the number of seconds from the start of the year to 2023-09-19 15:30:00
    """
    try:
        input_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        start_of_year = datetime(input_date.year, 1, 1)
        time_difference = input_date - start_of_year
        seconds = time_difference.total_seconds()
        return int(seconds)  # Convert to integer and return

    except ValueError:
        return "Invalid date format. Please use 'yyyy-mm-dd hh:mm:ss'."


class CorraiModel(Model):
    def __init__(
        self,
        model_path,
        model_name=None,
        output_list=None,
        build_profiling: str = None,
        omc_build_flags: str = None,
    ):
        self.model_path = model_path
        self.output_list = output_list
        self.session = Session(log_level=OMLogLevel.STATS)
        self.session.__enter__()
        self.session.build_model(
            model_path, profiling=build_profiling, omc_build_flags=omc_build_flags
        )

        if model_name is None:
            self.model_name = list(self.session._model_parameters.keys())[0]

        self.simulation_path = Path(
            self.session.get_binary_location(self.model_name)
        ).parent

    def __del__(self):
        self.session.__exit__()

    def get_parameters(self):
        return self.session.get_parameters()

    def set_simulation_options(self, simulation_options: dict, session=None):
        """
        simulation options keys must be:
            - solver
            - start: 'yyyy-mm-dd hh:mm:ss'
            - end: 'yyyy-mm-dd hh:mm:ss'
            - timestep: int [seconds]
        """

        if session is None:
            session = self.session

        try:
            session.set_solver(getattr(Solver, simulation_options["solver"].upper()))
        except KeyError:
            pass

        try:
            start = datetime.strptime(simulation_options["start"], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(simulation_options["end"], "%Y-%m-%d %H:%M:%S")
            sim_len = (end - start).total_seconds()
            start_sec = seconds_since_start_of_year(simulation_options["start"])

            session.set_time_range(
                start_time=int(start_sec),
                stop_time=int(start_sec + sim_len),
            )

        except KeyError:
            if "start" not in list(simulation_options.keys()) or "end" not in list(
                simulation_options.keys()
            ):
                raise ValueError(
                    "Cannot provide one of 'startTime' or 'stopTime' both shall be "
                    "provided at once"
                )
            else:
                pass

        try:
            session.set_tolerance(tolerance=simulation_options["tolerance"])
        except KeyError:
            pass

        try:
            for model in session._simulation_opts:
                session._simulation_opts[model].set_option(
                    "stepSize", simulation_options["timestep"]
                )
        except KeyError:
            pass

    def set_combi_time_table_df(self, df, combi_time_table_name, session=None):
        # DataFrame columns order must match the order
        # defined in the modelica file. This cannot be checked

        if session is None:
            session = self.session

        new_bounds_path = self.simulation_path / "bounds.txt"
        df_to_combitimetable(df, new_bounds_path)
        session.set_parameter(
            f"{combi_time_table_name}.fileName", new_bounds_path.as_posix()
        )

    def set_param_dict(self, param_dict: dict = None, session=None):
        if session is None:
            session = self.session

        if param_dict is not None:
            for key, value in param_dict.items():
                session.set_parameter(key, value)

    def get_results(self, model_name=None, reference_year: int = 2009, session=None):
        # Modelica solver can provide several results for one timestep
        # Moreover variable timestep solver can provide messy result
        # Manipulations are done to resample the index and provide seconds
        if session is None:
            session = self.session

        if model_name is None:
            model_name = self.model_name

        # A bit hacky. Get the last simulation in _solution dictionary
        # Should correspond to the last calculated result
        res = pd.DataFrame(
            session._solutions[model_name].get_solutions()[
                list(session._solutions[model_name].get_solutions().keys())[-1]
            ]
        )

        res.index = res["time"]
        res = res.drop("time", axis=1)

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{session._simulation_opts[self.model_name].get('stepSize')}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()

        res.index = seconds_to_datetime(res.index, reference_year)

        if self.output_list is None:
            return res
        else:
            return res[self.output_list]

    def evaluate(self, model_name: str = None, session=None):
        if session is None:
            session = self.session

        if model_name is None:
            model_name = self.model_name
        session.simulate(model_name=model_name)

    def simulate(
        self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        session = deepcopy(self.session)
        self.set_param_dict(parameter_dict, session=session)
        self.set_simulation_options(simulation_options, session=session)
        if simulation_options is None:
            ref_year = 2009
        else:
            try:
                date_parts = simulation_options["start"].split("-")
                ref_year = int(date_parts[0])
            except KeyError:
                raise ValueError("simulation_options dict must contain a 'start' key")

        self.evaluate(session=session)
        return self.get_results(reference_year=ref_year, session=session)


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
            self.set_combi_time_table_df(boundary_df, "Boundaries")
            if year is not None:
                warnings.warn(
                    "Simulator year is read from boundary"
                    "DataFrame. Argument year is ignored"
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

        self.simulation_path = Path(
            self.session.get_binary_location(self.model_name)
        ).parent

        if simulation_options is not None:
            self.set_simulation_options(simulation_options)

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

    def set_simulation_options(self, simulation_options: dict):
        try:
            self.session.set_solver(
                getattr(Solver, simulation_options["solver"].upper())
            )
            self.simulation_options["solver"] = simulation_options["solver"]
        except KeyError:
            pass

        try:
            self.session.set_time_range(
                start_time=simulation_options["startTime"],
                stop_time=simulation_options["stopTime"],
            )
            self.simulation_options["startTime"] = simulation_options["startTime"]
            self.simulation_options["stopTime"] = simulation_options["stopTime"]

        except KeyError:
            if "startTime" not in list(
                simulation_options.keys()
            ) or "stopTime" not in list(simulation_options.keys()):
                raise ValueError(
                    "Cannot provide one of 'startTime' or 'stopTime' both shall be "
                    "provided at once"
                )
            else:
                pass

        try:
            self.session.set_tolerance(tolerance=simulation_options["tolerance"])
            self.simulation_options["tolerance"] = simulation_options["tolerance"]
        except KeyError:
            pass

        try:
            for model in self.session._simulation_opts:
                self.session._simulation_opts[model].set_option(
                    "stepSize", simulation_options["stepSize"]
                )
                self.simulation_options["stepSize"] = simulation_options["stepSize"]
        except KeyError:
            pass

    def set_combi_time_table_df(self, df, combi_time_table_name):
        # DataFrame columns order must match the order
        # defined in the modelica file. This cannot be checked
        # Modelica file must contain a combiTimetable named Boundaries

        new_bounds_path = self.simulation_path / "bounds.txt"
        df_to_combitimetable(df, new_bounds_path)
        self.session.set_parameter(
            f"{combi_time_table_name}.fileName", new_bounds_path.as_posix()
        )
        try:
            self.year = df.index[0].year
        except ValueError:
            raise ValueError(
                "Could not read date from boundary condition. "
                "Please verify that Dataframe has DateTime index"
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

        # A bit hacky. Get the last simulation in _solution dictionary
        # Should correspond to the last calculated result
        res = pd.DataFrame(
            self.session._solutions[model_name].get_solutions()[
                list(self.session._solutions[model_name].get_solutions().keys())[-1]
            ]
        )

        res.index = res["time"]
        res = res.drop("time", axis=1)

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{self.session._simulation_opts[self.model_name].get('stepSize')}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()

        if not index_datetime:
            res.index = res.index.astype("int")
        else:
            res.index = seconds_to_datetime(res.index, self.year)

        if self.output_list is None:
            return res
        else:
            return res[self.output_list]
