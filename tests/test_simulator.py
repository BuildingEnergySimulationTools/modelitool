from pathlib import Path

import pytest

from modelitool.simulate import Simulator


import pandas as pd

PACKAGE_DIR = Path(__file__).parent / "TestLib"

SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl",
    'outputFormat': 'csv'
}

OUTPUTS = ['res.showNumber']


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    test_run_path = tmp_path_factory.mktemp("run")
    simu = Simulator(
        model_path="TestLib.rosen",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=SIMULATION_OPTIONS,
        output_list=OUTPUTS,
        simulation_path=test_run_path,
        lmodel=["Modelica"],
    )
    return simu


@pytest.fixture(scope="session")
def simul_none_run_path():
    simu = Simulator(
        model_path="TestLib.rosen",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=SIMULATION_OPTIONS,
        output_list=OUTPUTS,
        simulation_path=None,
        lmodel=["Modelica"],
    )
    return simu


@pytest.fixture(scope="session")
def simul_boundaries():
    simul_options = {
        "startTime": 16675200,
        "stopTime": 16682400,
        "stepSize": 1 * 3600,
        "tolerance": 1e-06,
        "solver": "dassl",
        'outputFormat': 'csv'
    }

    simu = Simulator(
        model_path="TestLib.boundary_test",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=simul_options,
        output_list=["Boundaries.y[1]", "Boundaries.y[2]"],
        simulation_path=None,
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

    def test_simulate_get_results(self, simul):
        simul.simulate()
        res = simul.get_results(index_datetime=False)
        ref = pd.DataFrame({"res.showNumber": [401.0, 401.0, 401.0]})
        assert ref.equals(res)

    def test_run_in_temp_dir(self, simul_none_run_path):
        # TODO Test do not remove created tempdir
        ref_path = simul_none_run_path._simulation_path.as_posix()
        sim_path = simul_none_run_path.omc.sendExpression("cd()")

        assert ref_path == sim_path

    def test_set_boundaries_df(self, simul_boundaries):
        new_bounds = pd.DataFrame(
            {"Boundaries.y[1]": [10, 20, 30], "Boundaries.y[2]": [3, 4, 5]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="H"),
        )
        new_bounds.index.freq = None
        new_bounds = new_bounds.astype(float)

        simul_boundaries.set_boundaries_df(new_bounds)
        simul_boundaries.simulate()
        res = simul_boundaries.get_results()

        pd.testing.assert_frame_equal(new_bounds, res)
