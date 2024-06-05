import os
import tempfile
from pathlib import Path

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
        simulation_options: dict[str, float | str | int] = None,
        x: pd.DataFrame = None,
        output_list: list[str] = None,
        simulation_path: Path = None,
        x_combitimetable_name: str = None,
        package_path: Path = None,
        lmodel: list[str] = None,
    ):
        """
        A class to wrap ompython to simulate Modelica system.
        Make it easier to change parameters values and simulation options.
        Allows specification of boundary conditions using Pandas Dataframe.
        The class inherits from corrai Model base class, and can be used with the
        module.

        - model_path (Path | str): Path to the Modelica model file.
        - simulation_options (dict[str, float | str | int], optional):
            Options for the simulation. May include values for "startTime",
            "stopTime", "stepSize", "tolerance", "solver", "outputFormat".
        - x (pd.DataFrame, optional): Input data for the simulation. Index shall
            be a DatetimeIndex or integers. Columns must match the combi time table
            used to specify boundary conditions in the Modelica System.
        - output_list (list[str], optional): List of output variables. Default
            will output all available variables.
        - simulation_path (Path, optional): Path to run the simulation and
            save the simulation results.
        - x_combitimetable_name (str, optional): Name of the Modelica System
            combi timetable object name, that is used to set the boundary condition.
        - package_path (Path, optional): Path to the Modelica package directory
            if necessary (package.mo).
        - lmodel (list[str], optional): List of Modelica libraries to load.
        """

        self.x_combitimetable_name = (
            x_combitimetable_name if x_combitimetable_name is not None else "Boundaries"
        )
        self._simulation_path = (
            simulation_path if simulation_path is not None else Path(tempfile.mkdtemp())
        )

        if not os.path.exists(self._simulation_path):
            os.mkdir(simulation_path)

        self._x = x if x is not None else pd.DataFrame()
        self.output_list = output_list
        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{self._simulation_path.as_posix()}")')

        model_system_args = {
            "fileName": (package_path or model_path).as_posix(),
            "modelName": model_path.stem if package_path is None else model_path,
            "lmodel": lmodel,
            "variableFilter": ".*" if output_list is None else "|".join(output_list),
        }

        self.model = ModelicaSystem(**model_system_args)
        if simulation_options is not None:
            self._set_simulation_options(simulation_options)

    def simulate(
        self,
        parameter_dict: dict = None,
        simulation_options: dict = None,
        x: pd.DataFrame = None,
        verbose: bool = True,
        simflags: str = None,
        year: int = None,
    ) -> pd.DataFrame:
        """
        Runs the simulation with the provided parameters, simulation options and
        boundariy conditions.
        - parameter_dict (dict, optional): Dictionary of parameters.
        - simulation_options (dict, optional): Will update simulation options if it
            had been given at the init phase. May include values for "startTime",
            "stopTime", "stepSize", "tolerance", "solver", "outputFormat".
        - x (pd.DataFrame, optional): Input data for the simulation. Index shall
            be a DatetimeIndex or integers. Columns must match the combi time table
            used to specify boundary conditions in the Modelica System.
        - verbose (bool, optional): If True, prints simulation progress. Defaults to
            True.
        - simflags (str, optional): Additional simulation flags.
        - year (int, optional): If x boundary conditions is not specified or do not
            have a DateTime index (seconds int), a year can be specified to convert
            int seconds index to a datetime index. If simulation spans overs several
            years, it shall be the year when it begins.
        """

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
            simflags=simflags,
            verbose=verbose,
        )

        if output_format == "csv":
            res = pd.read_csv(self._simulation_path / "res.csv", index_col=0)
            if self.output_list is not None:
                res = res.loc[:, self.output_list]
        else:
            if self.output_list is None:
                var_list = list(self.model.getSolutions())
            else:
                var_list = ["time"] + self.output_list

            res = pd.DataFrame(
                data=self.model.getSolutions(
                    varList=var_list,
                    resultfile=(self._simulation_path / result_file).as_posix(),
                ).T,
                columns=var_list,
            )

            res.set_index("time", inplace=True)

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(
            f"{int(self.model.getSimulationOptions()['stepSize'])}s"
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
            self.simulate(verbose=False)

        return list(self.model.getSolutions())

    def get_parameters(self):
        """
        Get parameters of the model or a loaded library.
        Returns:
            dict: Dictionary containing the parameters.
        """
        return self.model.getParameters()

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
            full_path = (self._simulation_path / "boundaries.txt").resolve().as_posix()
            self._set_param_dict({f"{self.x_combitimetable_name}.fileName": full_path})
            self._x = df

    def _set_param_dict(self, param_dict):
        self.model.setParameters([f"{item}={val}" for item, val in param_dict.items()])


def load_library(lib_path):
    """
    Load a Modelica library.

    Args:
        lib_path (str | Path): Path to the library directory.

    Returns:
        ModelicaSystem: An instance of ModelicaSystem if the library is loaded
        successfully.

    Raises:
        ValueError: If the library directory is not found.
    """
    if isinstance(lib_path, str):
        lib_path = Path(lib_path)

    if not lib_path.exists() or not lib_path.is_dir():
        raise ValueError(f"Library directory '{lib_path}' not found.")

    omc = OMCSessionZMQ()

    for root, _, files in os.walk(lib_path):
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

    for root, _, files in os.walk(library_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
