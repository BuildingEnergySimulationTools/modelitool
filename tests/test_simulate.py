from pathlib import Path

import pytest

from modelitool.simulate import Simulator, OMModel
from modelitool.simulate import load_library, library_contents
from tempfile import mkdtemp


import pandas as pd

PACKAGE_DIR = Path(__file__).parent / "TestLib"

SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl",
    "outputFormat": "csv",
}

OUTPUTS = ["res.showNumber"]


@pytest.fixture(scope="session")
def param_model():
    simu = Simulator(
        model_path="TestLib.paramModel",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=SIMULATION_OPTIONS,
        output_list=OUTPUTS,
        simulation_path=None,
        lmodel=["Modelica"],
    )
    return simu


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    test_run_path = tmp_path_factory.mktemp("run")
    simu = OMModel(
        model_path="TestLib.rosen",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=SIMULATION_OPTIONS,
        output_list=OUTPUTS,
        simulation_path=test_run_path,
        lmodel=["Modelica"],
    )
    return simu


class TestSimulator:
    def test_set_param_dict(self, simul):
        test_dict = {
            "x.k": 2.0,
            "y.k": 2.0,
        }

        simul._set_param_dict(test_dict)

        for key in test_dict.keys():
            assert float(test_dict[key]) == float(simul.model.getParameters()[key])

    def test_simulate_get_results(self, simul):
        res = simul.simulate()
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert ref.equals(res)

    def test_set_boundaries_df(self):
        simulation_options = {
            "startTime": 16675200,
            "stopTime": 16682400,
            "stepSize": 1 * 3600,
            "tolerance": 1e-06,
            "solver": "dassl",
            "outputFormat": "csv",
        }

        x = pd.DataFrame(
            {"Boundaries.y[1]": [10, 20, 30], "Boundaries.y[2]": [3, 4, 5]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="H"),
        )

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            output_list=["Boundaries.y[1]", "Boundaries.y[2]"],
            lmodel=["Modelica"],
        )

        res = simu.simulate(
            simulation_options=simulation_options,
            x=x
        )

        assert True
    def test_load_and_print_library(self, simul, capfd):
        libpath = PACKAGE_DIR
        try:
            load_library(libpath)
            assert True
        except ValueError:
            assert False, "library not loaded, failed test"

        library_contents(libpath)
        out, err = capfd.readouterr()
        assert "package.mo" in out

    def test_get_parameters(self, simul_param):
        param = simul_param.get_parameters()
        expected_param = {"k": "1.0"}
        assert param == expected_param
