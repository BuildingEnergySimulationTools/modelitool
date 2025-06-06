from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from modelitool.simulate import OMModel, library_contents, load_library

PACKAGE_DIR = Path(__file__).parent / "TestLib"


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    simulation_options = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl",
        "outputFormat": "csv",
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

        simul.set_param_dict(test_dict)

        for key in test_dict.keys():
            assert float(test_dict[key]) == float(simul.model.getParameters()[key])

        assert simul.get_parameters() == {
            "x.k": "2.0",
            "x.y": None,
            "y.k": "2.0",
            "y.y": None,
            "res.significantDigits": "2",
            "res.use_numberPort": "true",
        }

    def test_simulate_get_results(self, simul):
        assert simul.get_available_outputs() == [
            "time",
            "res.numberPort",
            "res.showNumber",
        ]
        res = simul.simulate()
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert ref.equals(res)

    def test_load_and_print_library(self, simul, capfd):
        libpath = PACKAGE_DIR
        try:
            load_library(libpath)
            assert True
        except ValueError as exc:
            raise AssertionError("library not loaded, failed test") from exc

        library_contents(libpath)
        out, err = capfd.readouterr()
        assert "package.mo" in out

    def test_get_parameters(self, simul):
        param = simul.get_parameters()
        expected_param = {
            "res.significantDigits": "2",
            "res.use_numberPort": "true",
            "x.k": "2.0",
            "x.y": None,
            "y.k": "2.0",
            "y.y": None,
        }
        assert param == expected_param

    def test_set_boundaries_df(self):
        simulation_options = {
            "startTime": 16675200,
            "stopTime": 16682400,
            "stepSize": 1 * 3600,
            "tolerance": 1e-06,
            "solver": "dassl",
            "outputFormat": "mat",
        }

        x_options = pd.DataFrame(
            {"Boundaries.y[1]": [10, 20, 30], "Boundaries.y[2]": [3, 4, 5]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
        )
        x_direct = pd.DataFrame(
            {"Boundaries.y[1]": [100, 200, 300], "Boundaries.y[2]": [30, 40, 50]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
        )

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            lmodel=["Modelica"],
        )

        simulation_options_with_x = simulation_options.copy()
        simulation_options_with_x["x"] = x_options
        res1 = simu.simulate(simulation_options=simulation_options_with_x)
        res1 = res1.loc[:, ["Boundaries.y[1]", "Boundaries.y[2]"]]
        np.testing.assert_allclose(x_options.to_numpy(), res1.to_numpy())
        assert np.all(
            [x_options.index[i] == res1.index[i] for i in range(len(x_options.index))]
        )
        assert np.all(
            [
                x_options.columns[i] == res1.columns[i]
                for i in range(len(x_options.columns))
            ]
        )

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            lmodel=["Modelica"],
        )
        res2 = simu.simulate(simulation_options=simulation_options, x=x_direct)
        res2 = res2.loc[:, ["Boundaries.y[1]", "Boundaries.y[2]"]]
        np.testing.assert_allclose(x_direct.to_numpy(), res2.to_numpy())
        assert np.all(
            [x_direct.index[i] == res2.index[i] for i in range(len(x_direct.index))]
        )
        assert np.all(
            [
                x_direct.columns[i] == res2.columns[i]
                for i in range(len(x_direct.columns))
            ]
        )

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            lmodel=["Modelica"],
        )
        with pytest.warns(
            UserWarning,
            match="Boundary file 'x' specified both in simulation_options and as a direct parameter",
        ):
            res3 = simu.simulate(
                simulation_options=simulation_options_with_x, x=x_direct
            )
            res3 = res3.loc[:, ["Boundaries.y[1]", "Boundaries.y[2]"]]
            np.testing.assert_allclose(x_direct.to_numpy(), res3.to_numpy())
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(x_options.to_numpy(), res3.to_numpy())
