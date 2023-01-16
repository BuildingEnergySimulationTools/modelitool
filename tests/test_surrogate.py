from pathlib import Path

import pandas as pd
import numpy as np

from modelitool.simulate import Simulator
from modelitool.surrogate import SimulationSampler
from modelitool.surrogate import get_aggregated_indicator
from modelitool.surrogate import SurrogateModel

from sklearn.metrics import mean_absolute_error

import pytest
import datetime as dt

PACKAGE_PATH = Path(__file__).parent / "TestLib/package.mo"

PARAM_DICT = {
    "x.k": [0, 10],
    "y.k": [0, 10]
}

SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl"
}

OUTPUTS = ["res.showNumber"]


@pytest.fixture(scope="session")
def rosen(tmp_path_factory):
    simu = Simulator(model_path="TestLib.rosen",
                     package_path=PACKAGE_PATH,
                     lmodel=["Modelica"],
                     simulation_options=SIMULATION_OPTIONS,
                     output_list=OUTPUTS)
    return simu


@pytest.fixture(scope="session")
def linear_2d(tmp_path_factory):
    simu = Simulator(model_path="TestLib.linear_2d",
                     package_path=PACKAGE_PATH,
                     lmodel=["Modelica"],
                     simulation_options=SIMULATION_OPTIONS,
                     output_list=OUTPUTS)
    return simu


class TestSurrogate:
    def test_simulation_sampler(self, rosen):
        sampler = SimulationSampler(
            simulator=rosen,
            parameters=PARAM_DICT,
        )

        sampler.add_sample(1, seed=42)

        ref = [26.75185347342908, 1.0, 10001.0, 1000081.0, 810081.0]

        to_test = [r.iloc[0, 0] for r in sampler.sample_results]

        assert to_test == ref

    def test_get_aggregated_indicator(self, rosen):
        sampler = SimulationSampler(
            simulator=rosen,
            parameters=PARAM_DICT,
        )

        sampler.add_sample(1, seed=42)

        y_array = get_aggregated_indicator(
            simulation_list=sampler.sample_results,
            indicator="res.showNumber",
            start=dt.datetime(2023, 1, 1, 0, 0, 1),
            end=dt.datetime(2023, 1, 1, 0, 0, 2),
            reference=pd.Series(
                data=[1., 1., 1.],
                index=pd.date_range("2023-01-01", freq='S', periods=3)
            ),
            method=mean_absolute_error
        )

        ref = np.array([
            2.575185e+01,
            0.000000e+00,
            1.000000e+04,
            1.000080e+06,
            8.100800e+05,
        ])

        assert np.allclose(y_array, ref, atol=1)

    def test_surrogate_model(self, linear_2d):
        surrogate = SurrogateModel(
            simulation_sampler=SimulationSampler(
                simulator=linear_2d,
                parameters=PARAM_DICT,
            ))

        surrogate.add_samples(100, seed=42)

        surrogate.fit_sample(
            indicator="res.showNumber",
            aggregation_method=np.mean,
        )

        res = surrogate.minimization_identification()

        assert np.allclose(
            res['x'], np.array([0., 0.]), atol=10E-3)
