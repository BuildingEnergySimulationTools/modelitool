import os
import warnings

import pandas as pd
import datetime as dt

from OMPython import ModelicaSystem
from OMPython import OMCSessionZMQ
import tempfile
from pathlib import Path

from corrai.base.model import Model

from modelitool.combitabconvert import df_to_combitimetable
from modelitool.combitabconvert import seconds_to_datetime
import re

# TODO Create a debug mode to print time


class OMModel(Model):
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
        self.loaded_libraries = {}

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
            self.run()

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

    def run(self, simflags=None):
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

    def simulate(
            self, parameter_dict: dict = None, simulation_options: dict = None
    ) -> pd.DataFrame:
        self.set_param_dict(parameter_dict)
        self.set_simulation_options(simulation_options)
        self.run()
        return self.get_results()


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

        self.library_path = None
        self.model_path = model_path if isinstance(model_path, Path) else Path(model_path)
        self.simulation_options = simulation_options
        self.output_list = output_list
        self.init_parameters = init_parameters or {}
        self.simulation_path = Path(tempfile.mkdtemp()) if simulation_path is None else Path(simulation_path)
        self.boundary_df = boundary_df
        self.year = year or dt.date.today().year
        self.package_path = package_path
        self.lmodel = lmodel

        self.simulation_path.mkdir(parents=True, exist_ok=True)

        self.omc = OMCSessionZMQ()

        self.loaded_libraries = {}

        if type(model_path) == str:
            model_path = Path(model_path)

        if simulation_path is None:
            simulation_path = tempfile.mkdtemp()
            simulation_path = Path(simulation_path)

        if not os.path.exists(simulation_path):
            os.mkdir(simulation_path)

        self.omc = OMCSessionZMQ()
        self.omc.sendExpression(f'cd("{simulation_path.as_posix()}")')

        self.model_path = model_path

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

    def load_library(self, lib_path):
        """
        Load a Modelica library.

        Args:
            lib_path (str): Path to the library directory.

        Returns:
            bool: True if the library is loaded successfully, False otherwise.
        """
        if isinstance(lib_path, str):
            lib_path = Path(lib_path)

        if not lib_path.exists() or not lib_path.is_dir():
            print(f"Library directory '{lib_path}' not found.")
            return False

        # Create an instance of ModelicaSystem for the library
        library_modelica_system = ModelicaSystem()

        # Connect to the OpenModelica session
        omc = OMCSessionZMQ()

        # Walk through the library directory
        for root, dirs, files in os.walk(lib_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".mo"):
                    # Load the Modelica file using the OMC session
                    omc.loadFile(file_path)

        # Store the ModelicaSystem instance in self.loaded_libraries
        library_name = lib_path.stem
        self.loaded_libraries[library_name] = library_modelica_system

        print(f"Library '{library_name}' loaded successfully.")

    def print_library_contents(self, library_path):
        """
        Print all files in the library recursively.

        Args:
            library_path (str): Path to the library directory.
        """
        for root, dirs, files in os.walk(library_path):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)

    def set_param_dict(
            self,
            parameters,
            library_path=None,
            package_name=None,
            model_name=None
    ):

        if package_name is None and model_name is None:
            self.model.setParameters([f"{item}={val}" for item, val in parameters.items()])

        else:
            lib_path = Path(library_path)
            self.library_path=lib_path
            package_path = os.path.join(self.library_path, package_name)
            model_file_path = os.path.join(package_path, f"{model_name}.mo")
            if not os.path.exists(model_file_path):
                print(f"File .mo for model {model_name} not found in library.")
                return

            with open(model_file_path, "r") as file:
                content = file.read()

            for param_name, param_value in parameters.items():
                pattern = re.compile(rf"(\b{re.escape(param_name)}\b\s*=\s*)([^;]*)")
                match = pattern.search(content)

                if match:
                    start, end = match.span()
                    content = content[:start] + f"{param_name} = {param_value}" + content[end:]

                else:
                    print(f"Parameter {param_name} not found in model {model_name}.")

            with open(model_file_path, "w") as file:
                file.write(content)

            # Restart of OMCSession
            self.omc = OMCSessionZMQ()
            self.__init__(
                model_path=self.model_path,
                simulation_options=self.simulation_options,
                output_list=self.output_list,
                init_parameters=getattr(self, 'init_parameters', None),
                simulation_path=getattr(self, '_simulation_path', None),
                boundary_df=getattr(self, 'boundary_df', None),
                year=getattr(self, 'year', None),
                package_path=getattr(self, 'package_path', None),
                lmodel=getattr(self, 'lmodel', [])
            )
