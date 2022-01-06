from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from modelitool.simulate import Simulator
from modelitool.probabilist import DCGenerator


class FakeSimulator:
    def __init__(self, output_list):
        self.output_list = output_list


@pytest.fixture()
def fake_simulator():
    return FakeSimulator(output_list=["x", 'y'])


class TestProbabilist:
    def test_dcgenerator_sample(self):
        test_params = {
            'a': (0, 1),
            'b': (2, 3),
            'c': (4, 5)
        }

        gen = DCGenerator(
            simulator=None,
            params=test_params,
            sample_size=3
        )

        expected = np.array([
            [0.541994, 2.541994, 4.541994],
            [0.619667, 2.619667, 4.619667],
            [0.057370, 2.057370, 4.057370],
        ])

        np.testing.assert_almost_equal(gen.sample, expected, decimal=6)

    def test_dcgenerator_get_dc(self, fake_simulator):
        test_params = {
            'a': (0, 1),
            'b': (2, 3),
            'c': (4, 5)
        }

        test_simu_results = np.array([
            [[0, 1],
             [1, 2]],
            [[2, 3],
             [3, 4]]
        ])

        test_observable_inputs = pd.DataFrame({
            "x1": [10, 11],
            "x2": [21, 22]
        })

        dc_gen = DCGenerator(
            simulator=fake_simulator,
            params=test_params,
            sample_size=2,
            observable_inputs=test_observable_inputs
        )

        dc_gen.simulation_results = test_simu_results

        expected_res = np.array([
            [0., 10., 21., 0.54199389, 2.54199389, 4.54199389],
            [1., 11., 22., 0.54199389, 2.54199389, 4.54199389],
            [2., 10., 21., 0.61966721, 2.61966721, 4.61966721],
            [3., 11., 22., 0.61966721, 2.61966721, 4.61966721]
        ])

        np.testing.assert_almost_equal(
            dc_gen.get_dc(indicator='x'), expected_res, decimal=6)
