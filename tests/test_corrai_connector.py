import numpy as np
import pandas as pd
from pathlib import Path

import pytest

from sklearn.metrics import mean_squared_error, mean_absolute_error
from modelitool.simulate import OMModel
from corrai.base.parameter import Parameter

from modelitool.corrai_connector import ModelicaFunction


PACKAGE_DIR = Path(__file__).parent / "TestLib"


PARAMETERS = [
    {Parameter.NAME: "x.k", Parameter.INTERVAL: (1.0, 3.0)},
    {Parameter.NAME: "y.k", Parameter.INTERVAL: (1.0, 3.0)},
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
            param_list=PARAMETERS,
            agg_methods_dict=agg_methods_dict,
            indicators=["res1.showNumber", "res2.showNumber"],
            reference_df=dataset,
            reference_dict=reference_dict,
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

    def test_custom_indicators(self, ommodel):
        mf = ModelicaFunction(
            om_model=ommodel,
            param_list=PARAMETERS,
            indicators=["res1.showNumber", "res2.showNumber", "custom_indicator"],
            custom_ind_dict={
                "custom_indicator": {
                    "depends_on": ["res1.showNumber", "res2.showNumber"],
                    "function": lambda x, y: x + y,
                }
            },
        )

        res = mf.function(X_DICT)

        # Test custom indicator
        np.testing.assert_allclose(
            res["custom_indicator"],
            expected_res["meas1"] + expected_res["meas2"],
            rtol=0.01,
        )

    def test_function_no_indicators(self, ommodel):
        mf = ModelicaFunction(
            om_model=ommodel,
            param_list=PARAMETERS,
            agg_methods_dict=None,
            indicators=None,
            reference_df=None,
            reference_dict=None,
        )

        res = mf.function(X_DICT)

        np.testing.assert_allclose(
            np.array([res["res1.showNumber"], res["res2.showNumber"]]),
            np.array([np.mean(expected_res["meas1"]), np.mean(expected_res["meas2"])]),
            rtol=0.01,
        )

    def test_warning_error(self, ommodel):
        # reference_df is not provided
        with pytest.raises(ValueError):
            ModelicaFunction(
                om_model=ommodel,
                param_list=PARAMETERS,
                reference_df=None,
                reference_dict=dataset,
            )

        # reference_dict is not provided
        with pytest.raises(ValueError):
            ModelicaFunction(
                om_model=ommodel,
                param_list=PARAMETERS,
                reference_df=dataset,
                reference_dict=None,
            )
