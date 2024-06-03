from pathlib import Path
import numpy as np
import pytest

from modelitool.simulate import Simulator, OMModel
from modelitool.simulate import load_library, library_contents


import pandas as pd

PACKAGE_DIR = Path(__file__).parent / "TestLib"


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    simulation_options = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl",
        "outputFormat": "mat",
    }

    outputs = ["res.showNumber"]

    test_run_path = tmp_path_factory.mktemp("run")
    simu = OMModel(
        model_path="TestLib.rosen",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=simulation_options,
        output_list=outputs,
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

    def test_get_parameters(self, simul):
        param = simul.get_parameters()
        expected_param = {'res.significantDigits': '2', 'res.use_numberPort': 'true', 'x.k': '2.0', 'x.y': None, 'y.k': '2.0', 'y.y': None}
        assert param == expected_param

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
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
        )

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            lmodel=["Modelica"],
        )

        res = simu.simulate(simulation_options=simulation_options, x=x)

        assert np.all([x.index[i] == res.index[i] for i in range(len(x.index))])
        np.testing.assert_allclose(x.to_numpy(), res.to_numpy())
        assert np.all([x.columns[i] == res.columns[i] for i in range(len(x.columns))])


