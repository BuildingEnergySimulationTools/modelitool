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
    Wrapper around OpenModelica (via OMPython) implementing the corrai Model API.

    This class encapsulates a Modelica model using ``OMPython.ModelicaSystem``
    and provides a stable Python interface to:

    - set Modelica parameters (``parameter`` variables),
    - configure and run simulations with custom simulation options,
    - retrieve simulation results as pandas DataFrames with a consistent time index.

    A key design constraint is that **Modelica parameters and simulation options
    are handled through distinct mechanisms**:

    - Model parameters are set using ``setParameters`` (via ``set_property_dict``).
    - Simulation options (startTime, stopTime, stepSize, solver, etc.) are set
      using ``setSimulationOptions`` and must NOT be passed as parameters.

    Parameters
    ----------
    model_path : Path or str
        Path to the Modelica model file (``.mo``) or to the top-level model
        when used with ``package_path``.
    output_list : list of str, optional
        List of Modelica variables to record during simulation.
        If None, all available outputs are kept.
    simulation_dir : Path, optional
        Directory where simulation files (results, override files, etc.)
        are written. If None, a temporary directory is created.
    boundary_table_name : str or None, optional
        Name of a ``CombiTimeTable`` instance in the Modelica model used to
        provide boundary conditions. If provided, boundary data can be passed
        via ``simulation_options["boundary"]``.
    package_path : Path, optional
        Path to a Modelica package directory (containing ``package.mo``).
        If provided, the model is loaded from the package instead of a single file.
    lmodel : list of str, optional
        List of Modelica libraries to load before instantiating the model.

    Methods
    -------
    simulate(property_dict=None, simulation_options=None, ...)
        Run a simulation with optional parameter overrides and simulation options.
        Returns a pandas DataFrame indexed by time.

    set_property_dict(property_dict)
        Set Modelica parameters (``parameter`` variables only).
        Simulation options are explicitly filtered out.

    get_property_dict()
        Return all current Modelica parameters with cleaned values.

    get_property_values(property_list)
        Return selected Modelica parameter values with cleaned values.

    Time handling
    -------------
    - Internally, OpenModelica simulations always run in seconds.
    - If ``startTime`` is provided as a ``pd.Timestamp`` or ``datetime``,
      the resulting DataFrame index is converted back to a timezone-aware
      ``DatetimeIndex`` anchored at the provided start time.
    - Solver-internal intermediate steps may generate irregular timestamps;
      duplicated timestamps are handled according to ``solver_duplicated_keep``.

    Examples
    --------
    >>> model = OMModel("MyModel.mo", output_list=["y"])
    >>> model.set_property_dict({"k": 2.0})
    >>> sim_opt = {
    ...     "startTime": pd.Timestamp("2025-02-17 00:00:00"),
    ...     "stopTime": pd.Timestamp("2025-03-17 00:00:00"),
    ...     "stepSize": pd.Timedelta("1h"),
    ...     "solver": "dassl",
    ...     "outputFormat": "csv",
    ... }
    >>> res = model.simulate(simulation_options=sim_opt)
    >>> res.head()
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

        def _strip_tz(x):
            if isinstance(x, pd.Timestamp):
                return x.tz_localize(None) if x.tz is not None else x
            if isinstance(x, dt.datetime):
                return x.replace(tzinfo=None) if x.tzinfo is not None else x
            return x

        start_tz = start.tz if isinstance(start, pd.Timestamp) \
            else getattr(start, "tzinfo", None)

        start = _strip_tz(start)
        stop = _strip_tz(stop)

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

            if start_tz is not None:
                res.index = res.index.tz_localize(start_tz)

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
                values.append([x.strip() if isinstance(x, str) else x for x in v])
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
        return {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}

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

    modelica_path = lib_path.parent.as_posix()
    lib_name = lib_path.stem

    omc.sendExpression(f'setModelicaPath("{modelica_path}")')
    success = omc.sendExpression(f'loadModel({lib_name})')

    if not success:
        err = omc.sendExpression("getErrorString()")
        raise RuntimeError(f"Failed to load Modelica library:\n{err}")

    print(f"Library '{lib_name}' loaded successfully.")


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
