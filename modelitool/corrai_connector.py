import numpy as np
import pandas as pd

from corrai.base.parameter import Parameter

from modelitool.simulate import OMModel


class ModelicaFunction:
    """
    A class that defines a function based on a Modelitool Simulator.

    Args:
        om_model (object): A fully configured Modelitool Simulator object.
        param_list (list): A list of parameter defined as dictionaries. At least , each
            parameter dict must have the following keys : "names", "interval".
        indicators (list, optional): A list of indicators to be returned by the
            function. An indicator must be one of the Simulator outputs. If not
            provided, all indicators in the simulator's output list will be returned.
            Default is None.
        agg_methods_dict (dict, optional): A dictionary that maps indicator names to
            aggregation methods. Each aggregation method should be a function that takes
            an array of values and returns a single value. It can also be an error
            function that will return an error indicator between the indicator results
            and a reference array of values defined in reference_df.
            If not provided, the default aggregation method for each indicator is
            numpy.mean. Default is None.
        reference_dict (dict, optional): When using an error function as agg_method, a
            reference_dict must be used to map indicator names to reference indicator
            names. The specified reference name will be used to locate the value in
            reference_df.
            If provided, the function will compute each indicator's deviation from its
            reference indicator using the corresponding aggregation method.
            Default is None.
        reference_df (pandas.DataFrame, optional): A pandas DataFrame containing the
            reference values for each reference indicator specified in reference_dict.
            The DataFrame should have the same length as the simulation results.
            Default is None.
        custom_ind_dict (dict, optional): A dictionary that maps indicator names to
        custom indicator information. Each custom indicator information should be
        a dictionary containing the following keys:
            - "depends_on": A list of indicator names that the custom function
                depends on. They should be in output list of simulator
            - "function": A function that computes the custom indicator values based
                on the values of indicators specified in "depends_on".
            If provided, the function will calculate custom indicators in addition
            to regular indicators. Default is None.

    Returns:
        pandas.Series: A pandas Series containing the function results.
        The index is the indicator names and the values are the aggregated simulation
        results.

    Raises:
        ValueError: If reference_dict and reference_df are not both provided or both
        None.
    """

    def __init__(
        self,
        om_model: OMModel,
        param_list,
        indicators=None,
        agg_methods_dict=None,
        reference_dict=None,
        reference_df=None,
        custom_ind_dict=None,
    ):
        self.om_model = om_model
        self.param_list = param_list
        if indicators is None:
            self.indicators = om_model.get_available_outputs()
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
        self.custom_ind_dict = custom_ind_dict if custom_ind_dict is not None else []

    def function(self, x_dict):
        """
        Calculates the function values for the given input dictionary.

        Args:
        - x_dict (dict): A dictionary of input values.

        Returns:
        - res_series (Series): A pandas Series object containing
        the function values with function names as indices.
        """
        temp_dict = {
            param[Parameter.NAME]: x_dict[param[Parameter.NAME]]
            for param in self.param_list
        }
        self.om_model._set_param_dict(temp_dict)
        res = self.om_model.simulate()

        function_results = {}

        # Calculate regular indicators
        for ind in self.indicators:
            if ind in res:
                function_results[ind] = res[ind]

        # Calculate custom indicators
        for ind in self.indicators:
            if ind not in function_results and ind in self.custom_ind_dict:
                ind_info = self.custom_ind_dict[ind]
                if all(output in res for output in ind_info["depends_on"]):
                    custom_values = ind_info["function"](
                        *[res[output] for output in ind_info["depends_on"]]
                    )
                    function_results[ind] = custom_values

        # Aggregate the indicators
        for ind in self.indicators:
            if ind in function_results and ind in self.agg_methods_dict:
                if self.reference_dict and ind in self.reference_dict:
                    ref_values = self.reference_df[self.reference_dict[ind]]
                    function_results[ind] = self.agg_methods_dict[ind](
                        function_results[ind], ref_values
                    )

                else:
                    function_results[ind] = self.agg_methods_dict[ind](
                        function_results[ind]
                    )

        res_series = pd.Series(function_results, dtype="float64")
        return res_series
