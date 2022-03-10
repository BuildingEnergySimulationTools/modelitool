from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from modelitool.sensitivity import modelitool_to_salib_problem
from modelitool.sensitivity import SAnalysis
from modelitool.simulate import Simulator


def mean_error(res, ref):
    return np.mean(ref - res)


def sum_error(res, ref):
    return np.sum(ref - res)


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
                     year=2009,
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
    common_index = pd.date_range(
        '2009-01-01 00:00:00',
        freq="s",
        periods=3
    )

    return [
        pd.DataFrame({
            'res1.showNumber': [5.88213201] * 3,
            'res2.showNumber': [4.80257357] * 3,
        }, index=common_index),
        pd.DataFrame({
            'res1.showNumber': [8.15192598] * 3,
            'res2.showNumber': [12.31778589] * 3,
        }, index=common_index),
        pd.DataFrame({
            'res1.showNumber': [1.42359607] * 3,
            'res2.showNumber': [5.95605462] * 3,
        }, index=common_index),
    ]


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

    def test__compute_aggregated(
            self, simul, expected_res, sa_param_config):
        ref_agg = pd.Series(
            [5.152551355, 5.152551355, 5.152551355],
            index=pd.date_range('2009-01-01 00:00:00', freq="s", periods=3)
        )

        sa_object = SAnalysis(
            simulator=simul,
            sensitivity_method="Sobol",
            parameters_config=sa_param_config
        )

        sa_object.simulation_results = expected_res

        res1 = sa_object._compute_aggregated(
            aggregation_method=np.mean,
            indicator="res1.showNumber"
        )

        res2 = sa_object._compute_aggregated(
            aggregation_method=np.mean,
            indicator="res2.showNumber"
        )

        res_mean = sa_object._compute_aggregated(
            aggregation_method=mean_error,
            indicator="res1.showNumber",
            ref=ref_agg
        )

        res_freq = sa_object._compute_aggregated(
            aggregation_method=np.mean,
            indicator='res1.showNumber',
            freq='2S'
        )

        res_freq_ref = sa_object._compute_aggregated(
            aggregation_method=sum_error,
            indicator='res1.showNumber',
            ref=ref_agg,
            freq='2S'
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
            res_mean,
            np.array([-0.729580657, -2.999374628, 3.728955285]),
            decimal=6
        )

        np.testing.assert_array_almost_equal(
            res_freq_ref,
            np.array([[1.45916131, 0.729580655],
                      [5.99874924, 2.99937462],
                      [-7.4579106, -3.7289552849]]),
            decimal=6
        )

        np.testing.assert_array_almost_equal(
            res_freq,
            np.array([[5.88213201, 5.88213201],
                      [8.15192598, 8.15192598],
                      [1.42359607, 1.42359607]]),
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
                   arguments={"print_to_console": False})

        sa.analyze(indicator='res1.showNumber',
                   aggregation_method=np.mean,
                   freq="2S",
                   arguments={"print_to_console": True})

        df_to_compare = pd.DataFrame({
            key: res['S1']
            for key, res in sa.sensitivity_dynamic_results.items()
        }).T

        np.testing.assert_almost_equal(
            sa.sensitivity_results['S1'],
            np.array([0.26933607, 1.255609, -0.81162613]),
            decimal=3
        )

        np.testing.assert_almost_equal(
            df_to_compare.to_numpy(),
            np.array([[0.26933607, 1.255609, -0.81162613],
                      [0.26933607, 1.255609, -0.81162613]]),
            decimal=3
        )



