import numpy as np
import pandas as pd
from pathlib import Path
from modelitool.functiongenerator import ModelicaFunction
from modelitool.simulate import Simulator
from modelitool.combitabconvert import seconds_to_datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

PACKAGE_DIR = Path(__file__).parent / "TestLib"

outputs = ["res1.showNumber", "res2.showNumber"]

simu_options = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl"
}

parameters = [
    {'name': 'x.k',
     'interval': (1.0, 3.0)},
    {'name': 'y.k',
     'interval': (1.0, 3.0)},
]

agg_methods_dict = {
    "res1.showNumber": mean_squared_error,
    "res2.showNumber": mean_absolute_error
}

reference_dict = {
    "res1.showNumber": "meas1",
    "res2.showNumber": "meas2"
}


class TestModelicaFunction:

    def test_function(self):
        simu_options = {
            "startTime": 0,
            "stopTime": 1,
            "stepSize": 1,
            "tolerance": 1e-06,
            "solver": "dassl"
        }

        simu = Simulator(model_path="TestLib.ishigami_two_outputs",
                         package_path=PACKAGE_DIR / "package.mo",
                         simulation_options=simu_options,
                         output_list=outputs,
                         simulation_path=None,
                         lmodel=["Modelica"])

        dataset = pd.DataFrame(
            {
                "meas1": [6, 2],
                "meas2": [14, 1],
            },
            index=pd.date_range('2023-01-01 00:00:00', freq="s", periods=2))

        expected_res = pd.DataFrame(
            {
                "meas1": [8.15, 8.15],
                "meas2": [12.31, 12.31],
            },
            index=pd.date_range('2023-01-01 00:00:00', freq="s", periods=2))

        mf = ModelicaFunction(simulator=simu,
                              param_dict=parameters,
                              agg_methods_dict=agg_methods_dict,
                              indicators=["res1.showNumber", "res2.showNumber"],
                              reference_df=dataset,
                              reference_dict=reference_dict)
        x_dict = {
            'x.k': 2,
            'y.k': 2
        }

        res = mf.function(x_dict)

        np.testing.assert_allclose(
            np.array([res["res1.showNumber"], res["res2.showNumber"]]),
            np.array([mean_squared_error(expected_res['meas1'], dataset['meas1']),
                      mean_absolute_error(expected_res['meas2'], dataset['meas2'])]),
            rtol=0.01)

