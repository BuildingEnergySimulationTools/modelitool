from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from corrai.base.parameter import Parameter

from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelitool.corrai_connector import ModelicaFunction
from modelitool.simulate import OMModel

PACKAGE_DIR = Path(__file__).parent / "TestLib"


PARAMETERS = [
    Parameter(name= "x.k", interval= (1.0, 3.0)),
    Parameter(name= "y.k", interval= (1.0, 3.0)),
]

agg_methods_dict = {
    "res1.showNumber": mean_squared_error,
    "res2.showNumber": mean_absolute_error,
}

reference_dict = {"res1.showNumber": "meas1", "res2.showNumber": "meas2"}


X_DICT = {"x.k": 2, "y.k": 2}

dataset = pd.DataFrame(
    {
        "meas1": [6, 2],
        "meas2": [14, 1],
    },
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)

expected_res = pd.DataFrame(
    {
        "meas1": [8.15, 8.15],
        "meas2": [12.31, 12.31],
    },
    index=pd.date_range("2023-01-01 00:00:00", freq="s", periods=2),
)


@pytest.fixture(scope="session")
def ommodel(tmp_path_factory):
    simu_options = {
        "startTime": 0,
        "stopTime": 1,
        "stepSize": 1,
        "tolerance": 1e-06,
        "solver": "dassl",
        "outputFormat": "csv",
    }

    outputs = ["res1.showNumber", "res2.showNumber"]

    simu = OMModel(
        model_path="TestLib.ishigami_two_outputs",
        package_path=PACKAGE_DIR / "package.mo",
        simulation_options=simu_options,
        output_list=outputs,
        lmodel=["Modelica"],
    )

    return simu


class TestModelicaFunction:
    def test_function_indicators(self, ommodel):
        mf = ModelicaFunction(
            om_model=ommodel,
            parameters=PARAMETERS,
            indicators_config={
                "res1.showNumber": ( mean_squared_error, dataset["meas1"]),
                "res2.showNumber": ( mean_absolute_error, dataset["meas2"]),
            },
            scipy_obj_indicator=["res1.showNumber", "res2.showNumber"],
        )

        res = mf.function(X_DICT)

        np.testing.assert_allclose(
            np.array([res["res1.showNumber"], res["res2.showNumber"]]),
            np.array(
                [
                    mean_squared_error(expected_res["meas1"], dataset["meas1"]),
                    mean_absolute_error(expected_res["meas2"], dataset["meas2"]),
                ]
            ),
            rtol=0.01,
        )

    def test_scipy_obj_function_and_bounds(self, ommodel):
        mf = ModelicaFunction(
            om_model=ommodel,
            parameters=PARAMETERS,
            indicators_config={"res1.showNumber": (mean_squared_error, dataset["meas1"])},
            scipy_obj_indicator="res1.showNumber",
        )

        val1 = mf.scipy_obj_function([2.0, 2.0])
        assert isinstance(val1, float)
        with pytest.raises(ValueError):
            mf.scipy_obj_function([1.0])
        mf.scipy_obj_indicator = "unknown"
        with pytest.raises(KeyError):
            mf.scipy_obj_function([2.0, 2.0])

        bnds = mf.bounds
        assert bnds == [(1.0, 3.0), (1.0, 3.0)]

    def test_init_values(self, ommodel):
        params_with_init = [
            Parameter(name="x.k", interval=(0, 1), init_value=0.5),
            Parameter(name="y.k", interval=(1, 2), init_value=1.5),
        ]
        mf = ModelicaFunction(
            om_model=ommodel,
            parameters=params_with_init,
            indicators_config={"res1.showNumber": (mean_squared_error, dataset["meas1"])},
        )
        assert mf.init_values == [0.5, 1.5]

        params_without_init = [
            Parameter(name="x.k", interval=(0, 1)),
            Parameter(name="y.k", interval=(1, 2)),
        ]
        mf2 = ModelicaFunction(
            om_model=ommodel,
            parameters=params_without_init,
            indicators_config={"res1.showNumber": (mean_squared_error, dataset["meas1"])},
        )
        assert mf2.init_values is None