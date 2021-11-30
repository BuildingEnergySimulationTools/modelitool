from pathlib import Path

import numpy as np

import pytest
from modelitool.sensitivity import modelitool_to_salib_problem
from modelitool.sensitivity import SAnalysis
from modelitool.simulate import Simulator


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    curr_mod_path = Path(__file__).parent / "modelica/ishigami_two_outputs.mo"

    test_run_path = tmp_path_factory.mktemp("run")
    simulation_opt = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl"
    }

    outputs = ["res1.showNumber", "res2.showNumber"]

    simu = Simulator(model_path=curr_mod_path,
                     simulation_path=test_run_path,
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


class TestSensitivity:
    def test_modelitool_to_salib_problem(self):
        sa_problem = {
            'num_vars': 5,
            'names': [
                'f_stegos.k',
                'R_stegos.R',
                'alpha_alu',
                'R_al21',
                'C_al',
            ],
            'bounds': [
                [0, 0.2],
                [0.023 - 0.023 * 0.3, 0.023 + 0.023 * 0.3],
                [0.2 - 0.2 * 0.3, 0.2 + 0.2 * 0.3],
                [0.00005 - 0.00005 * 0.3, 0.00005 + 0.00005 * 0.3],
                [2700 - 2700 * 0.3, 2700 + 2700 * 0.3]
            ]
        }

        as_config = {
            'f_stegos.k': [0, 0.2],
            'R_stegos.R': [0.023 - 0.023 * 0.3, 0.023 + 0.023 * 0.3],
            'alpha_alu': [0.2 - 0.2 * 0.3, 0.2 + 0.2 * 0.3],
            'R_al21': [0.00005 - 0.00005 * 0.3, 0.00005 + 0.00005 * 0.3],
            'C_al': [2700 - 2700 * 0.3, 2700 + 2700 * 0.3],
        }

        assert modelitool_to_salib_problem(as_config) == sa_problem

    def test_run_simulation(self, simul):
        params = {
            "x.k": [1.0, 3.0],
            "y.k": [1.0, 3.0],
            "z.k": [1.0, 3.0],
        }

        sample = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0]])

        expected_res = np.array([
            [
                [5.88213201, 4.80257357],
                [5.88213201, 4.80257357],
                [5.88213201, 4.80257357],
            ],
            [
                [8.15192598, 12.31778589],
                [8.15192598, 12.31778589],
                [8.15192598, 12.31778589],
            ],
            [
                [1.42359607, 5.95605462],
                [1.42359607, 5.95605462],
                [1.42359607, 5.95605462],
            ]
        ])

        sa_object = SAnalysis(
            simulator=simul,
            sensitivity_method="Sobol",
            parameters_config=params
        )

        sa_object.sample = sample

        sa_object.run_simulations()

        print(sa_object.simulation_results)

        np.testing.assert_array_almost_equal(
            sa_object.simulation_results,
            expected_res,
            decimal=6
        )

