import datetime as dt
import os
import tempfile
import warnings
from pathlib import Path
import win32api

import pandas as pd
from OMPython import ModelicaSystem
from OMPython import OMCSessionZMQ
from corrai.base.model import Model

from modelitool.combitabconvert import df_to_combitimetable
from modelitool.combitabconvert import seconds_to_datetime


class OMModel(Model):
    def __init__(
            self,
            model_path: Path | str,
            simulation_options: dict[str, float | str | int]= None,
            x: pd.DataFrame = None,
            output_list: list[str] = None,
            simulation_path: Path = None,
            x_combitimetable_name: str = None,
            package_path: Path = None,
            lmodel: list[str] = None,
    ):
        self.x_combitimetable_name = (
            x_combitimetable_name if x_combitimetable_name is not None else "Boundaries"
        )

        if simulation_path is None:
            self._simulation_path = Path(tempfile.mkdtemp())
        else:
            self._simulation_path = simulation_path

        if not os.path.exists(self._simulation_path):
            os.mkdir(simulation_path)

        self._x = x if x is not None else pd.DataFrame()

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{self._simulation_path.as_posix()}")')

        lmodel = [] if lmodel is None else lmodel
        if package_path is None:
            model_system_args = {
                "fileName": model_path.as_posix(),
                "modelName": model_path.stem,
                "lmodel": lmodel,
                "variableFilter": "|".join(output_list),
            }
        else:
            model_system_args = {
                "fileName": package_path.as_posix(),
                "modelName": model_path,
                "lmodel": lmodel,
                "variableFilter": "|".join(output_list),
            }

        self.model = ModelicaSystem(**model_system_args)
        self.output_list = output_list
        if simulation_options is not None:
            self._set_simulation_options(simulation_options)

    def simulate(
            self,
            parameter_dict: dict = None,
            simulation_options: dict = None,
            x: pd.DataFrame = None,
            simflags=None,
            year: int = None
    ) -> pd.DataFrame:

        if parameter_dict is not None:
            self._set_param_dict(parameter_dict)

        if simulation_options is not None:
            self._set_simulation_options(simulation_options)

        if x is not None:
            self._set_x(x)

        output_format = self.model.getSimulationOptions()["outputFormat"]
        result_file = "res.csv" if output_format == "csv" else "res.mat"
        self.model.simulate(
            resultfile=(self._simulation_path / result_file).as_posix(),
            simflags=simflags
        )

        if output_format == "csv":
            res = pd.read_csv(self._simulation_path / "res.csv", index_col=0)
            res = res.loc[:, self.output_list]
        else:
            sol_list = self.model.getSolutions(
                ["time"] + self.output_list, resultfile="res.mat").T
            res = pd.DataFrame(
                sol_list[:, 1:],
                index=sol_list[:, 0].flatten(),
                columns=self.output_list,
            )

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{int(self.model.getSimulationOptions()['stepSize'])}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()

        if not self._x.empty:
            res.index = seconds_to_datetime(res.index, self._x.index[0].year)
        elif year is not None:
            res.index = seconds_to_datetime(res.index, year)
        else:
            res.index = res.index.astype("int")
        return res

    def save(self, file_path: Path):
        pass

    def get_available_outputs(self):
        if self.model.getSolutions() is None:
            # A bit dirty but simulation must be run once so
            # getSolutions() can access results
            self.simulate()

        return self.model.getSolutions()

    def _set_simulation_options(self, simulation_options):
        self.model.setSimulationOptions(
            [
                f'startTime={simulation_options["startTime"]}',
                f'stopTime={simulation_options["stopTime"]}',
                f'stepSize={simulation_options["stepSize"]}',
                f'tolerance={simulation_options["tolerance"]}',
                f'solver={simulation_options["solver"]}',
                f'outputFormat={simulation_options["outputFormat"]}',
            ]
        )
        self.simulation_options = simulation_options

    def _set_x(self, df: pd.DataFrame):
        """Sets the input data for the simulation and updates the corresponding file."""
        if not self._x.equals(df):
            new_bounds_path = self._simulation_path / "boundaries.txt"
            df_to_combitimetable(df, new_bounds_path)
            full_path = win32api.GetLongPathName((self._simulation_path / "boundaries.txt").as_posix())
            self._set_param_dict(
                {f"{self.x_combitimetable_name}.fileName": full_path}
            )
            self._x = df

    def _set_param_dict(self, param_dict):
        self.model.setParameters([f"{item}={val}" for item, val in param_dict.items()])


class Simulator:
    def __init__(
            self,
            model_path,
            simulation_options,
            output_list,
            init_parameters=None,
            simulation_path=None,
            boundary_df=None,
            year=None,
            package_path=None,
            lmodel=[],
    ):

        if type(model_path) == str:
            model_path = Path(model_path)

        if simulation_path is None:
            simulation_path = tempfile.mkdtemp()
            simulation_path = Path(simulation_path)

        if not os.path.exists(simulation_path):
            os.mkdir(simulation_path)

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{simulation_path.as_posix()}")')

        # A bit dirty but the only way I found to change the simulation dir
        # ModelicaSystem take cwd as currDirectory
        os.chdir(simulation_path)
        if package_path is None:
            model_system_args = {
                "fileName": model_path.as_posix(),
                "modelName": model_path.stem,
                "lmodel": lmodel,
                "variableFilter": "|".join(output_list),
            }
        else:
            model_system_args = {
                "fileName": package_path.as_posix(),
                "modelName": model_path,
                "lmodel": lmodel,
                "variableFilter": "|".join(output_list),
            }

        self.model = ModelicaSystem(**model_system_args)

        self._simulation_path = simulation_path
        self.output_list = output_list

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

        self.set_simulation_options(simulation_options)

    def get_available_outputs(self):
        if self.model.getSolutions() is None:
            # A bit dirty but simulation must be run once so
            # getSolutions() can access results
            self.simulate()

        return self.model.getSolutions()

    def set_simulation_options(self, simulation_options):
        self.model.setSimulationOptions(
            [
                f'startTime={simulation_options["startTime"]}',
                f'stopTime={simulation_options["stopTime"]}',
                f'stepSize={simulation_options["stepSize"]}',
                f'tolerance={simulation_options["tolerance"]}',
                f'solver={simulation_options["solver"]}',
                f'outputFormat={simulation_options["outputFormat"]}',
            ]
        )
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
        # t1 = time()
        self.model.setParameters([f"{item}={val}" for item, val in param_dict.items()])
        # t2 = time()
        # print(f"Setting new parameters took {t2-t1}s")

    def get_parameters(self):
        """
        Get parameters of the model or a loaded library.
        Returns:
            dict: Dictionary containing the parameters.
        """
        return self.model.getParameters()

    def simulate(self, simflags=None):
        self.simflags = simflags
        if self.simulation_options["outputFormat"] == "csv":
            resultfile = "res.csv"
        else:
            resultfile = "res.mat"
        self.model.simulate(resultfile=resultfile, simflags=simflags)
        self.resultfile = resultfile

    def get_results(self, index_datetime=True):
        # Modelica solver can provide several results for one timestep
        # Moreover variable timestep solver can provide messy result
        # Manipulations are done to resample the index and provide seconds

        if self.simulation_options["outputFormat"] == "csv":
            res = pd.read_csv(self._simulation_path / "res.csv", index_col=0)

        else:
            sol_list = self.model.getSolutions(
                ["time"] + self.output_list, resultfile="res.mat"
            ).T
            res = pd.DataFrame(
                sol_list[:, 1:],
                index=sol_list[:, 0].flatten(),
                columns=self.output_list,
            )
            res.columns = self.output_list

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{int(self.model.getSimulationOptions()['stepSize'])}S"
        ).mean()
        res.index = res.index.to_series().dt.total_seconds()

        if not index_datetime:
            res.index = res.index.astype("int")
        else:
            res.index = seconds_to_datetime(res.index, self.year)

        # t2 = time()
        # print(f"Getting results took {t2-t1}s")
        return res


def load_library(lib_path):
    """
    Load a Modelica library.

    Args:
        lib_path (str | Path): Path to the library directory.

    Returns:
        ModelicaSystem: An instance of ModelicaSystem if the library is loaded successfully.

    Raises:
        ValueError: If the library directory is not found.
    """
    if isinstance(lib_path, str):
        lib_path = Path(lib_path)

    if not lib_path.exists() or not lib_path.is_dir():
        raise ValueError(f"Library directory '{lib_path}' not found.")

    omc = OMCSessionZMQ()

    for root, dirs, files in os.walk(lib_path):
        for file in files:
            if file.endswith(".mo"):
                file_path = os.path.join(root, file)
                omc.sendExpression(f'loadFile("{file_path}")')

    print(f"Library '{lib_path.stem}' loaded successfully.")


def library_contents(library_path):
    """
    Print all files in the library recursively.

    Args:
        library_path (str | Path): Path to the library directory.
    """
    library_path = Path(library_path) if isinstance(library_path, str) else library_path

    if not library_path.exists() or not library_path.is_dir():
        raise ValueError(f"Library directory '{library_path}' not found.")

    for root, dirs, files in os.walk(library_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
