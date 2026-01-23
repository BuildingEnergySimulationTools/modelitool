from pathlib import Path

import pytest

import pandas as pd

from modelitool.simulate import OMModel, library_contents, load_library

PACKAGE_DIR = Path(__file__).parent / "TestLib"


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    test_run_path = tmp_path_factory.mktemp("run")
    simu = OMModel(
        model_path="TestLib.rosen",
        package_path=PACKAGE_DIR / "package.mo",
        output_list=["res.showNumber"],
        simulation_dir=test_run_path,
        lmodel=["Modelica"],
    )
    return simu


class TestSimulator:
    def test_get_property_values(self, simul):
        values = simul.get_property_values(["x.k", "y.k"])
        assert isinstance(values, list)
        assert len(values) == 2
        assert values[0], values[1] == ["2.0"]

        # Comment while ompython version < 4+
        # with pytest.raises(KeyError):
        #     simul.get_property_values("nonexistent.param")

    def test_set_param_dict_and_simulation_options(self, simul):
        # test change of parameters
        test_dict = {
            "x.k": 2.0,
            "y.k": 2.0,
        }

        simul.set_property_dict(test_dict)

        for key in test_dict.keys():
            assert float(test_dict[key]) == float(simul.model.getParameters()[key])

        assert simul.get_property_dict() == {
            "x.k": "2.0",
            "x.y": None,
            "y.k": "2.0",
            "y.y": None,
            "res.significantDigits": "2",
            "res.use_numberPort": "true",
        }

        # test change of parameters AND simulations options
        # because issues were found when both are changed
        # in overide.txt file
        res_dt = simul.simulate(
            simulation_options={
                "startTime": pd.Timestamp("2009-02-01 00:00:00", tz="UTC"),
                "stopTime": pd.Timestamp("2009-02-01 00:00:02", tz="UTC"),
                "stepSize": pd.Timedelta("1s"),
                "tolerance": 1e-06,
                "solver": "dassl",
                "outputFormat": "mat",
            }
        )

        assert res_dt.index[0] == pd.Timestamp("2009-02-01 00:00:00", tz="UTC")

        # get property value
        assert simul.get_property_values("x.k") == [["2.0"]]
        assert simul.get_property_values(["x.k", "y.k"]) == [["2.0"], ["2.0"]]

    def test_simulate_get_results(self, simul):
        simulation_options = {
            "startTime": 0,
            "stopTime": 2,
            "stepSize": 1,
            "tolerance": 1e-06,
            "solver": "dassl",
            "outputFormat": "csv",
        }

        res = simul.simulate(simulation_options=simulation_options)
        ref = pd.DataFrame({"res.showNumber": [401, 401, 401]})
        assert ref.equals(res)

        res_dt = simul.simulate(
            simulation_options={
                "startTime": pd.Timestamp("2009-01-01 00:00:00", tz="UTC"),
                "stopTime": pd.Timestamp("2009-01-01 00:00:02", tz="UTC"),
                "stepSize": pd.Timedelta("1s"),
                "tolerance": 1e-06,
                "solver": "dassl",
                "outputFormat": "mat",
            }
        )

        ref = pd.DataFrame(
            {"res.showNumber": [401.0, 401.0, 401.0]},
            pd.date_range(
                "2009-01-01 00:00:00", freq="s", periods=3, tz="UTC", name="time"
            ),
        )

        pd.testing.assert_frame_equal(res_dt, ref, check_freq=False)

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
        param = simul.get_property_dict()
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
        boundaries_seconds = pd.DataFrame(
            {"x1": [10, 20, 30], "x2": [3, 4, 5]},
            index=[16675200, 16678800, 16682400],
        )

        simulation_options = {
            "startTime": 16675200,
            "stopTime": 16682400,
            "stepSize": 3600,
            "tolerance": 1e-06,
            "solver": "dassl",
            "boundary": boundaries_seconds,
        }

        simu = OMModel(
            model_path="TestLib.boundary_test",
            package_path=PACKAGE_DIR / "package.mo",
            lmodel=["Modelica"],
            boundary_table_name="Boundaries",
        )

        res = simu.simulate(simulation_options=simulation_options)

        x_direct = pd.DataFrame(
            {"Boundaries.y[1]": [100, 200, 300], "Boundaries.y[2]": [30, 40, 50]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="h"),
        )
