from pathlib import Path

import pytest

from modelitool.simulate import Simulator
import pandas as pd

from time import time

@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    curr_mod_path = Path(__file__).parent / "modelica/rosen_out_txt.mo"

    test_run_path = tmp_path_factory.mktemp("run")
    simulation_opt = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl"
    }

    outputs = ["res.showNumber"]

    simu = Simulator(model_path=curr_mod_path,
                     simulation_path=test_run_path,
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


class TestSimulator:
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