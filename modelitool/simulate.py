import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem, OMCSessionZMQ

from corrai.base.model import Model

from modelitool.combitabconvert import df_to_combitimetable, seconds_to_datetime


class OMModel(Model):
    """
    Wrap OpenModelica (via OMPython) in the corrai Model formalism.

    Parameters
    ----------
    model_path : Path | str
        Path to the Modelica model file.
    simulation_options : dict, optional
        Dictionary of simulation options including:
        ``startTime``, ``stopTime``, ``stepSize``, ``tolerance``,
        ``solver``, ``outputFormat``.
        Can also include ``boundary`` (pd.DataFrame) if the model
        uses a CombiTimeTable.
    output_list : list of str, optional
        List of variables to record during simulation.
    simulation_path : Path, optional
        Directory where simulation files will be written.
    boundary_table : str or None, optional
        Name of the CombiTimeTable object in the Modelica model
        that is used to provide boundary conditions.

        - If a string is provided, boundary data can be passed through
          ``simulation_options["boundary"]``.
        - If None (default), no CombiTimeTable will be set and any
          provided ``boundary`` will be ignored.
    package_path : Path, optional
        Path to the Modelica package directory (package.mo).
    lmodel : list of str, optional
        List of Modelica libraries to load.

    Examples
    --------
    >>> import pandas as pd
    >>> from corrai.om import OMModel
    >>> model = OMModel("MyModel.mo", output_list=["y"], boundary_table="Boundaries")
    >>> x = pd.DataFrame({"y": [1, 2, 3]}, index=[0, 1, 2])
    >>> res = model.simulate(simulation_options={"boundary": x, "stepSize": 1})
    """

    def __init__(
        self,
        model_path: Path | str,
        simulation_options: dict[str, float | str | int] = None,
        output_list: list[str] = None,
        simulation_path: Path = None,
        boundary_table: str | None = None,
        package_path: Path = None,
        lmodel: list[str] = None,
        omhome: Path | str = None,
        is_dynamic=True,
    ):
        self.boundary_table = boundary_table
        self._simulation_path = (
            simulation_path if simulation_path is not None else Path(tempfile.mkdtemp())
        )
        self._x = pd.DataFrame()
        self.output_list = output_list

        if not os.path.exists(self._simulation_path):
            os.mkdir(self._simulation_path)

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{self._simulation_path.as_posix()}")')

        model_system_args = {
            "fileName": (package_path or model_path).as_posix(),
            "modelName": model_path.stem if package_path is None else model_path,
            "lmodel": lmodel if lmodel is not None else [],
            "variableFilter": ".*" if output_list is None else "|".join(output_list),
        }
        self.model = ModelicaSystem(**model_system_args)

        if simulation_options is not None:
            self.set_simulation_options(simulation_options)

        self.is_dynamic = is_dynamic


    def set_simulation_options(self, simulation_options: dict | None = None):
        if simulation_options is None:
            return

        if "boundary" in simulation_options:
            if self.boundary_table is None:
                warnings.warn(
                    "Boundary provided but no combitimetable name set -> ignoring.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.set_boundary(simulation_options["boundary"])

        standard_options = {
            "startTime": simulation_options.get("startTime"),
            "stopTime": simulation_options.get("stopTime"),
            "stepSize": simulation_options.get("stepSize"),
            "tolerance": simulation_options.get("tolerance"),
            "solver": simulation_options.get("solver"),
            "outputFormat": simulation_options.get("outputFormat"),
        }
        options = [f"{k}={v}" for k, v in standard_options.items() if v is not None]
        self.model.setSimulationOptions(options)
        self.simulation_options = simulation_options

    def set_boundary(self, df: pd.DataFrame):
        """Set boundary data and update parameters accordingly."""
        if not self._x.equals(df):
            new_bounds_path = self._simulation_path / "boundaries.txt"
            df_to_combitimetable(df, new_bounds_path)
            full_path = new_bounds_path.resolve().as_posix()
            self.set_param_dict({f"{self.boundary_table}.fileName": full_path})
            self._x = df

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
        simflags: str = None,
        year: int = None,
    ) -> pd.DataFrame:
        if property_dict is not None:
            self.set_param_dict(property_dict)

        self.set_simulation_options(simulation_options)

        output_format = self.model.getSimulationOptions()["outputFormat"]
        result_file = "res.csv" if output_format == "csv" else "res.mat"
        self.model.simulate(
            resultfile=(self._simulation_path / result_file).as_posix(),
            simflags=simflags,
        )

        if output_format == "csv":
            res = pd.read_csv(self._simulation_path / "res.csv", index_col=0)
            if self.output_list is not None:
                res = res.loc[:, self.output_list]
        else:
            var_list = ["time"] + (self.output_list or list(self.model.getSolutions()))
            raw = self.model.getSolutions(
                varList=var_list,
                resultfile=(self._simulation_path / result_file).as_posix(),
            )

            arr = np.atleast_2d(raw).T

            _, unique_idx = np.unique(var_list, return_index=True)
            var_list = [var_list[i] for i in sorted(unique_idx)]
            arr = arr[:, sorted(unique_idx)]

            res = pd.DataFrame(arr, columns=var_list)
            res.set_index("time", inplace=True)

        res.index = pd.to_timedelta(res.index, unit="second")
        res = res.resample(f"{int(self.model.getSimulationOptions()['stepSize'])}s").mean()
        res.index = res.index.to_series().dt.total_seconds()

        if not self._x.empty:
            res.index = seconds_to_datetime(res.index, self._x.index[0].year)
        elif year is not None:
            res.index = seconds_to_datetime(res.index, year)
        else:
            res.index = res.index.astype("int")

        return res


    def get_property_values(
        self, property_list: str | tuple[str, ...] | list[str]
    ) -> list[str | int | float | None]:
        if isinstance(property_list, str):
            property_list = (property_list,)
        return [self.model.getParameters(prop) for prop in property_list]

    def get_available_outputs(self):
        if self.model.getSolutions() is None:
            self.simulate(verbose=False)
        return list(self.model.getSolutions())

    def get_parameters(self):
        return self.model.getParameters()

    def set_param_dict(self, param_dict):
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
