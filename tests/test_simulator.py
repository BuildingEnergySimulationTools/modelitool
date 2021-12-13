from pathlib import Path

import pytest

from modelitool.simulate import Simulator
from modelitool.simulate import seconds_to_datetime

from datetime import timedelta

import pandas as pd


MODEL_PATH = Path(__file__).parent / "modelica/rosen.mo"

SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl"
}

OUTPUTS = ["res.showNumber"]


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    test_run_path = tmp_path_factory.mktemp("run")
    simu = Simulator(model_path=MODEL_PATH,
                     simulation_options=SIMULATION_OPTIONS,
                     output_list=OUTPUTS,
                     simulation_path=test_run_path)
    return simu


@pytest.fixture(scope="session")
def simul_none_run_path():
    simu = Simulator(model_path=MODEL_PATH,
                     simulation_options=SIMULATION_OPTIONS,
                     output_list=OUTPUTS,
                     simulation_path=None)
    return simu


class TestSimulator:

    def test_seconds_to_datetime(self):
        test_index = pd.Series([
            timedelta(seconds=43200).total_seconds(),
            timedelta(seconds=43500).total_seconds()
        ])

        expected_res = pd.to_datetime([
            "2009-01-01 12:00:00",
            "2009-01-01 12:05:00"
        ])

        res = seconds_to_datetime(test_index, 2009)

        pd.testing.assert_index_equal(expected_res, res)

    def test_set_param_dict(self, simul):
        test_dict = {
            "x.k": 2.0,
            "y.k": 2.0,
        }

        simul.set_param_dict(test_dict)

        for key in test_dict.keys():
            assert float(test_dict[key]) == float(
                simul.model.getParameters()[key]
            )

    def test_simulate_get_results(self, simul):
        simul.simulate()

        res = simul.get_results()

        ref = pd.DataFrame({
            "res.showNumber": [401.0, 401.0, 401.0]
        })

        assert ref.equals(res)

    def test_run_in_temp_dir(self, simul_none_run_path):
        # TODO Test do not remove created tempdir
        ref_path = simul_none_run_path._simulation_path.as_posix()
        sim_path = simul_none_run_path.omc.sendExpression('cd()')

        assert ref_path == sim_path

