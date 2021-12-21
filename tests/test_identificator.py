from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from modelitool.combitabconvert import seconds_to_datetime
from modelitool.simulate import Simulator
from modelitool.identify import Identificator

PACKAGE_PATH = Path(__file__).parent / "TestLib/package.mo"

ID_PARAMS = {
    'x.k': {
        "init": 5,
        "interval": (0, 10)
    },
    'y.k': {
        "init": 5,
        "interval": (0, 10)
    },
}

DATASET = pd.DataFrame(
    {
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [3, 4, 5, 7, 8, 9],
        "y": [5, 8, 11, 15, 18, 21]
    },
    index=seconds_to_datetime(
        pd.Index([0, 1, 2, 10, 11, 12]).astype(float),
        ref_year=2009)
)


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


@pytest.fixture(scope="session")
def simul_linear(tmp_path_factory):
    simulation_opt = {
        "startTime": 0,
        "stopTime": 2,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl"
    }

    outputs = ["res.showNumber"]

    simu = Simulator(model_path="TestLib.linear_two_dimension",
                     package_path=PACKAGE_PATH,
                     lmodel=["Modelica"],
                     simulation_options=simulation_opt,
                     output_list=outputs)
    return simu


class TestIdentificator:
    def test_fit_default(self, simul):
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
        )

        ident.fit(features=None, labels=np.array([0, 0, 0]))

        np.testing.assert_allclose(
            np.array(list(ident.param_identified.values())),
            np.array([1., 1.]), rtol=0.01
        )

    def test_fit_features(self, simul_linear):

        ident = Identificator(
            simulator=simul_linear,
            parameters=ID_PARAMS,
        )

        ident.fit(
            features=DATASET[["x1", "x2"]].head(3),
            labels=DATASET[["y"]].head(3),
        )

        print("hello")

        np.testing.assert_allclose(
            np.array(list(ident.param_identified.values())),
            np.array([2., 1.]), rtol=0.01
        )