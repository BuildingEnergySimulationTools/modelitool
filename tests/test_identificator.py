from pathlib import Path

import pytest
import numpy as np

from modelitool.simulate import Simulator
from modelitool.identify import Identificator


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

    outputs = ["res.showNumber"]

    simu = Simulator(model_path="TestLib.rosen",
                     package_path=package_path,
                     lmodel=["Modelica"],
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


class TestIdentificator:
    def test_simulate_get_results(self, simul):
        id_params = {
            'x.k': {
                "init": 5,
                "interval": (-5, 5)
            },
            'y.k': {
                "init": 5,
                "interval": (-5, 5)
            },
        }

        ident = Identificator(
            simulator=simul,
            parameters=id_params,
            y_train=np.array([0, 0, 0])
        )

        res = ident.fit()

        np.testing.assert_allclose(
            res.x, np.array([1., 1.]), rtol=0.01
        )
