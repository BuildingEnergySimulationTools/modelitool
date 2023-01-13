from pathlib import Path

from modelitool.simulate import Simulator
from modelitool.surrogate import SimulationSampler

import pytest

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
