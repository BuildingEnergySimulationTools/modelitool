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
        "time_index": "seconds",
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
    def test_get_property_values(self, simul):
        values = simul.get_property_values(["x.k", "y.k"])
        assert isinstance(values, list)
        assert len(values) == 2
        assert values[0],  values[1] == ["2.0"]

        with pytest.raises(KeyError):
            simul.get_property_values("nonexistent.param")

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

    def test_simulate_time_index_modes(self, simul):
        res = simul.simulate()
        assert isinstance(res.index[0], (int, np.integer))

        res_dt = simul.simulate(simulation_options={
            "startTime": 0,
            "stopTime": 2,
            "stepSize": 1,
            "tolerance": 1e-06,
            "solver": "dassl",
            "outputFormat": "csv",
            "time_index": "datetime",
        })
        assert isinstance(res_dt.index, pd.DatetimeIndex)

        res_year = simul.simulate(simulation_options={
            "startTime": 0,
            "stopTime": 2,
            "stepSize": 1,
            "tolerance": 1e-06,
            "solver": "dassl",
            "outputFormat": "csv",
            "ref_year": 2023,
        })
        assert isinstance(res_year.index, pd.DatetimeIndex)
        assert res_year.index[0].year == 2023

# TODO to be fixed with new version of OMPYTHON
#     def test_set_boundaries_df(self):
#         simulation_options = {
#             "startTime": 16675200,
#             "stopTime": 16682400,
#             "stepSize": 1 * 3600,
#             "tolerance": 1e-06,
#             "solver": "dassl",
#             "outputFormat": "csv",
#         }
#
#         x_options = pd.DataFrame(
#             {"Boundaries.y[1]": [10, 20, 30], "Boundaries.y[2]": [3, 4, 5]},
#             index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
#         )
#         x_direct = pd.DataFrame(
#             {"Boundaries.y[1]": [100, 200, 300], "Boundaries.y[2]": [30, 40, 50]},
#             index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
#         )
#
#         simu = OMModel(
#             model_path="TestLib.boundary_test",
#             package_path=PACKAGE_DIR / "package.mo",
#             lmodel=["Modelica"],
#             boundary_table="Boundaries",
#         )
#
#         simulation_options_with_boundary = simulation_options.copy()
#         simulation_options_with_boundary["boundary"] = x_options
#         res1 = simu.simulate(simulation_options=simulation_options_with_boundary)
#         res1 = res1.loc[:, ["Boundaries.y[1]", "Boundaries.y[2]"]]
#         np.testing.assert_allclose(x_options.to_numpy(), res1.to_numpy())
#         assert all(x_options.index == res1.index)
#         assert all(x_options.columns == res1.columns)
#
#         simu = OMModel(
#             model_path="TestLib.boundary_test",
#             package_path=PACKAGE_DIR / "package.mo",
#             lmodel=["Modelica"],
#             boundary_table="Boundaries",
#         )
#         simulation_options_with_boundary = simulation_options.copy()
#         simulation_options_with_boundary["boundary"] = x_direct
#         res2 = simu.simulate(simulation_options=simulation_options_with_boundary)
#         res2 = res2.loc[:, ["Boundaries.y[1]", "Boundaries.y[2]"]]
#         np.testing.assert_allclose(x_direct.to_numpy(), res2.to_numpy())
#         assert all(x_direct.index == res2.index)
#         assert all(x_direct.columns == res2.columns)
#
#         simu = OMModel(
#             model_path="TestLib.boundary_test",
#             package_path=PACKAGE_DIR / "package.mo",
#             lmodel=["Modelica"],
#             boundary_table=None,
#         )
#         with pytest.warns(UserWarning, match="Boundary provided but no combitimetable name set"):
#             simu.simulate(simulation_options=simulation_options_with_boundary)
