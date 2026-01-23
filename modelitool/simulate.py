import datetime as dt
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem, OMCSessionZMQ

from corrai.base.model import Model
from corrai.fmu import (
    datetime_index_to_seconds_index,
    parse_simulation_times,
    seconds_index_to_datetime_index,
)

from sklearn.pipeline import Pipeline

from modelitool.combitabconvert import write_combitt_from_df

DEFAULT_SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 24 * 3600,
    "stepSize": 60,
    "solver": "dassl",
    "tolerance": 1e-6,
    "outputFormat": "mat",
}


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
        output_list: list[str] = None,
        simulation_dir: Path = None,
        boundary_table_name: str | None = None,
        package_path: Path = None,
        lmodel: list[str] = None,
    ):
        super().__init__(is_dynamic=True)
        self.boundary_table_name = boundary_table_name
        self.output_list = output_list

        self.simulation_dir = (
            Path(tempfile.mkdtemp()) if simulation_dir is None else simulation_dir
        )

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{self.simulation_dir.as_posix()}")')

        model_system_args = {
            "fileName": (package_path or model_path).as_posix(),
            "modelName": model_path.stem if package_path is None else model_path,
            "lmodel": lmodel if lmodel is not None else [],
            "variableFilter": ".*" if output_list is None else "|".join(output_list),
        }
        self.model = ModelicaSystem(**model_system_args)
        self.property_dict = self.get_property_dict()

    def simulate(
        self,
        property_dict: dict[str, str | int | float] = None,
        simulation_options: dict = None,
            solver_duplicated_keep: str = "last",
            post_process_pipeline: Pipeline = None,
        simflags: str = None,
    ) -> pd.DataFrame:
        """
        Run an OpenModelica simulation and return results as a pandas DataFrame.

        Parameters
        ----------
        property_dict : dict, optional
            Dictionary of model parameters to update before simulation.
            Keys must match Modelica parameter names.
        simulation_options : dict, optional
            Simulation options in the same format as in ``OMModel.__init__``.
            If ``simulation_options["boundary"]`` is provided and the model has
            a ``boundary_table`` name, the DataFrame is exported as a
            CombiTimeTable-compatible text file and injected into the model.
        simflags : str, optional
            Additional simulator flags passed directly to OpenModelica.

        Returns
        -------
        pandas.DataFrame
            Simulation results indexed either by:

            - a timestamp index if a boundary table is used
              (the year is inferred from ``boundary.index[0].year``), or
            - integer seconds since the simulation start otherwise.

            The DataFrame columns include either:
            - the variables listed in ``output_list``, or
            - all variables produced by OpenModelica.

        """

        simu_property = self.property_dict.copy()
        simu_property.update(dict(property_dict or {}))

        simulation_options = {
            **DEFAULT_SIMULATION_OPTIONS,
            **(simulation_options or {}),
        }

        start, stop, step = (
            simulation_options.get(it, None)
            for it in ["startTime", "stopTime", "stepSize"]
        )

        # Output step cannot be used in ompython
        start_sec, stop_sec, step_sec, _ = parse_simulation_times(
            start, stop, step, step
        )
        om_simu_opt = simulation_options | {
            "startTime": start_sec,
            "stopTime": stop_sec,
            "stepSize": step_sec,
        }

        boundary_df = None
        if simu_property:
            boundary_df = simu_property.pop("boundary", boundary_df)

        if simulation_options:
            sim_boundary = om_simu_opt.pop("boundary", boundary_df)

            if boundary_df is None and sim_boundary is not None:
                boundary_df = sim_boundary
            elif boundary_df is not None and sim_boundary is not None:
                warnings.warn(
                    "Boundary specified in both property_dict and "
                    "simulation_options. The one in property_dict will be used.",
                    UserWarning,
                    stacklevel=2,
                )

        if boundary_df is not None:
            boundary_df = boundary_df.copy()
            if isinstance(boundary_df.index, pd.DatetimeIndex):
                boundary_df.index = datetime_index_to_seconds_index(boundary_df.index)

            if not (
                boundary_df.index[0] <= start_sec <= boundary_df.index[-1]
                and boundary_df.index[0] <= stop_sec <= boundary_df.index[-1]
            ):
                raise ValueError(
                    "'startTime' and 'stopTime' are outside boundary DataFrame"
                )

            write_combitt_from_df(boundary_df, self.simulation_dir / "boundaries.txt")
            full_path = (self.simulation_dir / "boundaries.txt").resolve().as_posix()
            self.set_property_dict({f"{self.boundary_table_name}.fileName": full_path})


        if property_dict is not None:
            self.set_property_dict(property_dict)

        self.model.setSimulationOptions(om_simu_opt)

        output_format = self.model.getSimulationOptions()["outputFormat"]
        result_file = "res.csv" if output_format == "csv" else "res.mat"
        self.model.simulate(
            resultfile=(self.simulation_dir / result_file).as_posix(),
            simflags=simflags,
        )

        if output_format == "csv":
            res = pd.read_csv(self.simulation_dir / "res.csv", index_col=0)
            if self.output_list is not None:
                res = res.loc[:, self.output_list]
        else:
            var_list = ["time"] + (self.output_list or list(self.model.getSolutions()))
            raw = self.model.getSolutions(
                varList=var_list,
                resultfile=(self.simulation_dir / result_file).as_posix(),
            )

            arr = np.atleast_2d(raw).T

            _, unique_idx = np.unique(var_list, return_index=True)
            var_list = [var_list[i] for i in sorted(unique_idx)]
            arr = arr[:, sorted(unique_idx)]

            res = pd.DataFrame(arr, columns=var_list).set_index("time")

        if isinstance(start, (pd.Timestamp, dt.datetime)):
            res.index = seconds_index_to_datetime_index(res.index, start.year)
            res.index = res.index.round("s")
            res = res.tz_localize(start.tz)
            res.index.freq = res.index.inferred_freq
        else:
            res.index = round(res.index.to_series(), 2)

        res = res.loc[~res.index.duplicated(keep=solver_duplicated_keep)]
        if post_process_pipeline is not None:
            res = post_process_pipeline.fit_transform(res)

        return res

    def get_property_values(
            self, property_list: str | tuple[str, ...] | list[str]
    ) -> list[list[str | int | float | None]]:
        if isinstance(property_list, str):
            property_list = (property_list,)

        values = []
        for prop in property_list:
            v = self.model.getParameters(prop)
            if isinstance(v, list):
                values.append([
                    x.strip() if isinstance(x, str) else x
                    for x in v
                ])
            else:
                values.append(v.strip() if isinstance(v, str) else v)

        return values

    # TODO Find a way to get output without simulation
    # def get_available_outputs(self):
    #     try:
    #         sols = self.model.getSolutions()
    #     except ModelicaSystemError:
    #         self.simulate()
    #         sols = self.model.getSolutions()
    #     return list(sols)

    def get_property_dict(self):
        raw = self.model.getParameters()
        return {
            k: (v.strip() if isinstance(v, str) else v)
            for k, v in raw.items()
        }

    def set_property_dict(self, property_dict):
        self.model.setParameters(
            [f"{item}={val}\n" for item, val in property_dict.items()]
        )

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
