from pathlib import Path

import pandas as pd
import numpy as np

from modelitool.simulate import Simulator
from modelitool.surrogate import SimulationSampler
from modelitool.surrogate import get_aggregated_indicator

from sklearn.metrics import mean_absolute_error

import pytest
import datetime as dt

PACKAGE_PATH = Path(__file__).parent / "TestLib/package.mo"

PARAM_DICT = {
    "x.k": [0, 10],
    "y.k": [0, 10]
}


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    simulation_opt = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl"
    }

    outputs = ["res.showNumber"]

    simu = Simulator(model_path="TestLib.rosen",
                     package_path=PACKAGE_PATH,
                     lmodel=["Modelica"],
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


class TestSurrogate:
    def test_simulation_sampler(self, simul):
        sampler = SimulationSampler(
            simulator=simul,
            parameters=PARAM_DICT,
        )

        sampler.add_sample(1, seed=42)

        ref = [26.75185347342908, 1.0, 10001.0, 1000081.0, 810081.0]

        to_test = [r.iloc[0, 0] for r in sampler.sample_results]

        assert to_test == ref

    def test_get_aggregated_indicator(self, simul):
        sampler = SimulationSampler(
            simulator=simul,
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
