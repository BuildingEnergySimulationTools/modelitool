import numpy as np
import pandas as pd


class ModelicaFunction:
    def __init__(
        self,
        simulator,
        param_dict,
        indicators=None,
        agg_methods_dict=None,
        reference_dict=None,
        reference_df=None,
    ):
        self.simulator = simulator
        self.param_dict = param_dict
        if indicators is None:
            self.indicators = simulator.output_list
        else:
            self.indicators = indicators
        if agg_methods_dict is None:
            self.agg_methods_dict = {ind: np.mean for ind in self.indicators}
        else:
            self.agg_methods_dict = agg_methods_dict
        if (reference_dict is not None and reference_df is None) or (
            reference_dict is None and reference_df is not None
        ):
            raise ValueError("Both reference_dict and reference_df should be provided")
        self.reference_dict = reference_dict
        self.reference_df = reference_df

    def function(self, x_dict):
        temp_dict = {param["name"]: x_dict[param["name"]] for param in self.param_dict}
        self.simulator.set_param_dict(temp_dict)
        self.simulator.simulate()
        res = self.simulator.get_results()

        res_series = pd.Series(dtype="float64")
        solo_ind_names = self.indicators
        if self.reference_dict is not None:
            for k in self.reference_dict.keys():
                res_series[k] = self.agg_methods_dict[k](
                    res[k], self.reference_df[self.reference_dict[k]]
                )

            solo_ind_names = [
                i for i in self.indicators if i not in self.reference_dict.keys()
            ]

        for ind in solo_ind_names:
            res_series[ind] = self.agg_methods_dict[ind](res[ind])

        return res_series
