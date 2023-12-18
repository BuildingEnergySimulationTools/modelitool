from pathlib import Path

import pytest

from modelitool.simulate import Simulator, seconds_since_start_of_year, CorraiModel
from corrai.base.simulate import run_models_in_parallel


import pandas as pd

PACKAGE_DIR = Path(__file__).parent / "TestLib"

SIMULATION_OPTIONS = {
    "startTime": 0,
    "stopTime": 2,
    "stepSize": 1,
    "tolerance": 1e-06,
    "solver": "dassl",
}

OUTPUTS = ["res.showNumber"]


@pytest.fixture(scope="session")
def simul(tmp_path_factory):
    simu = Simulator(
        model_path=PACKAGE_DIR / "rosen.mo",
        simulation_options=SIMULATION_OPTIONS,
        output_list=OUTPUTS,
    )
    return simu


@pytest.fixture(scope="session")
def simul_boundaries():
    simul_options = {
        "startTime": 16675200,
        "stopTime": 16682400,
        "stepSize": 1 * 3600,
        "tolerance": 1e-06,
        "solver": "dassl",
    }

    simu = Simulator(
        model_path=PACKAGE_DIR / "boundary_test.mo",
        simulation_options=simul_options,
        output_list=["Boundaries.y[1]", "Boundaries.y[2]"]
    )
    return simu


class TestSimulator:
    def test_corrai_model(self):
        simu_opt = dict(
            start="2009-12-31 22:00:00", end="2010-01-01 02:00:00", timestep=3600
        )

        params = pd.DataFrame({"x.k": [0.0, 1.0, 2.0], "y.k": [0.0, 1.0, 2.0]})

        corrai_model = CorraiModel(
            model_path=PACKAGE_DIR / "linear_2d.mo", output_list=["res.showNumber"]
        )

        # res = corrai_model.simulate(parameter_dict=params, simulation_options=simu_opt)
        res = run_models_in_parallel(
            corrai_model, parameter_samples=params, simulation_options=simu_opt, n_cpu=3
        )
        assert True

    def test_seconds_since_start_of_year(self):
        assert seconds_since_start_of_year("2009-01-01 00:00:00") == 0

    def test_set_param_dict(self, simul):
        test_dict = {
            "x.k": 3.0,
            "y.k": 3.0,
        }

        simul.set_param_dict(test_dict)

        simul.simulate()

        for key in test_dict.keys():
            assert test_dict[key] == simul.session.get_parameters()[key]["value"]

    def test_simulate_get_results(self, simul):
        simul.set_param_dict({"x.k": 3.0, "y.k": 3.0})

        simul.simulate()

        res = simul.get_results(index_datetime=False)

        ref = pd.DataFrame({"res.showNumber": [3604.0, 3604.0, 3604.0]})

        assert ref.equals(res)

    def test_set_boundaries_df(self, simul_boundaries):
        new_bounds = pd.DataFrame(
            {"Boundaries.y[1]": [10, 20, 30], "Boundaries.y[2]": [3, 4, 5]},
            index=pd.date_range("2009-07-13 00:00:00", periods=3, freq="H"),
        )
        new_bounds.index.freq = None
        new_bounds = new_bounds.astype(float)
        new_bounds.index.names = ["time"]

        simul_boundaries.set_combi_time_table_df(new_bounds, "Boundaries")
        simul_boundaries.simulate()
        res = simul_boundaries.get_results()

        pd.testing.assert_frame_equal(new_bounds, res)
