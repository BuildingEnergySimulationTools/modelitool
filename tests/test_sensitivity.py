from pathlib import Path

import numpy as np

import pytest
from modelitool.sensitivity import modelitool_to_salib_problem
from modelitool.sensitivity import SAnalysis
from modelitool.simulate import Simulator


def mean_error(res, ref):
    return np.mean(ref - res)


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    package_path = Path(__file__).parent / "TestLib/package.mo"

    simulation_opt = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl"
    }

    outputs = ["res1.showNumber", "res2.showNumber"]

    simu = Simulator(model_path="TestLib.ishigami_two_outputs",
                     package_path=package_path,
                     lmodel=["Modelica"],
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


@pytest.fixture()
def sa_param_config():
    return {
        "x.k": [1.0, 3.0],
        "y.k": [1.0, 3.0],
        "z.k": [1.0, 3.0],
    }


@pytest.fixture()
def expected_res():
    return np.array([
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


class TestSensitivity:
    def test_modelitool_to_salib_problem(self, sa_param_config):
        sa_problem = {
            'num_vars': 3,
            'names': [
                'x.k',
                'y.k',
                'z.k',
            ],
            'bounds': [
                [1.0, 3.0],
                [1.0, 3.0],
                [1.0, 3.0],
            ]
        }

        print(modelitool_to_salib_problem(sa_param_config))

        assert modelitool_to_salib_problem(sa_param_config) == sa_problem

    def test_run_simulation(self, simul, sa_param_config, expected_res):
        test_sample = np.array([[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0]])

        sa_object = SAnalysis(
            simulator=simul,
            sensitivity_method="Sobol",
            parameters_config=sa_param_config
        )

        sa_object.sample = test_sample

        sa_object.run_simulations()

        np.testing.assert_array_almost_equal(
            sa_object.simulation_results,
            expected_res,
            decimal=6
        )

    def test_get_indicator_from_simulation_results(
            self, simul, expected_res, sa_param_config):
        sa_object = SAnalysis(
            simulator=simul,
            sensitivity_method="Sobol",
            parameters_config=sa_param_config
        )

        sa_object.simulation_results = expected_res

        res1 = sa_object.get_indicator_from_simulation_results(
            aggregation_method=np.mean,
            indicator="res1.showNumber"
        )

        res2 = sa_object.get_indicator_from_simulation_results(
            aggregation_method=np.mean,
            indicator="res2.showNumber"
        )

        res_ref = sa_object.get_indicator_from_simulation_results(
            aggregation_method=mean_error,
            indicator="res1.showNumber",
            ref=np.array([5.152551355, 5.152551355, 5.152551355])
        )

        np.testing.assert_array_almost_equal(
            res1,
            np.array([5.88213201, 8.15192598, 1.42359607]),
            decimal=6
        )

        np.testing.assert_array_almost_equal(
            res2,
            np.array([4.802573569, 12.31778589, 5.956054618]),
            decimal=6
        )

        np.testing.assert_array_almost_equal(
            res_ref,
            np.array([-0.729580657, -2.999374628, 3.728955285]),
            decimal=6
        )

    def test_analyse(self, simul):
        modelitool_problem = {
            "x.k": [-3.14159265359, 3.14159265359],
            "y.k": [-3.14159265359, 3.14159265359],
            "z.k": [-3.14159265359, 3.14159265359]
        }

        sa = SAnalysis(
            simulator=simul,
            sensitivity_method="Sobol",
            parameters_config=modelitool_problem
        )

        sa.draw_sample(n=1)

        sa.run_simulations()

        sa.analyze(indicator='res1.showNumber',
                   aggregation_method=np.mean,
                   arguments={"print_to_console": True})

        np.testing.assert_almost_equal(
            sa.sensitivity_results['S1'],
            np.array([0.26933607, 1.255609 ,-0.81162613]),
            decimal=3
        )

