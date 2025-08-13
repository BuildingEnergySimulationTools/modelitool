from typing import Callable, Iterable
import numpy as np
import pandas as pd

from corrai.base.parameter import Parameter
from modelitool.simulate import OMModel


class ModelicaFunction:
    """
    Objective-like wrapper around a Modelitool `OMModel` to compute
    aggregated indicators for calibration / optimisation, with the same
    ergonomics as `ObjectiveFunction`.

    Parameters
    ----------
    om_model : OMModel
        A configured Modelitool simulator (must expose `set_param_dict` and `simulate`).
    parameters : list[Parameter]
        Parameter definitions (name, interval/values, model_property, etc.).
    indicators_config : dict[str, Callable | tuple[Callable, pd.Series | pd.DataFrame | None]]
        For each indicator (i.e. a column returned by the simulation),
        either:
          - an aggregation function, e.g. np.mean, np.sum, custom metric; or
          - a tuple (func, reference) if the function requires a reference
            (e.g. sklearn.metrics.mean_squared_error).
    simulation_options : dict | None, default None
        Stored for consistency with ObjectiveFunction. Not directly passed to OMModel
        (which usually reads its own inputs), but kept here if you want to align APIs.
    scipy_obj_indicator : str | None, default None
        Which indicator to use as scalar objective for `scipy_obj_function`.
        Defaults to the first key of `indicators_config`.

    Notes
    -----
    - Parameter values are converted to a `property_dict` using `Parameter.model_property`
      when provided; otherwise the `Parameter.name` is used.
    - If `model_property` is a tuple of paths, the same scalar value is assigned to each path.
    """

    def __init__(
        self,
        om_model: OMModel,
        parameters: list[Parameter],
        indicators_config: dict[str, Callable | tuple[Callable, pd.Series | pd.DataFrame | None]],
        simulation_options: dict | None = None,
        scipy_obj_indicator: str | None = None,
    ):
        self.om_model = om_model
        self.parameters = list(parameters)
        self.indicators_config = dict(indicators_config)
        self.simulation_options = {} if simulation_options is None else simulation_options
        self.scipy_obj_indicator = (
            next(iter(self.indicators_config)) if scipy_obj_indicator is None else scipy_obj_indicator
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """List of (low, high) bounds for Real/Integer parameters with intervals."""
        bnds: list[tuple[float, float]] = []
        for p in self.parameters:
            if p.interval is None:
                raise ValueError(
                    f"Parameter {p.name!r} has no 'interval'; cannot expose numeric bounds."
                )
            lo, hi = p.interval
            bnds.append((float(lo), float(hi)))
        return bnds

    @property
    def init_values(self) -> list[float] | None:
        """Initial values if every parameter defines `init_value`, else None."""
        if all(p.init_value is not None for p in self.parameters):
            vals: list[float] = []
            for p in self.parameters:
                iv = p.init_value
                if isinstance(iv, (list, tuple)):
                    vals.append(float(iv[0]))
                else:
                    vals.append(float(iv))  # type: ignore[arg-type]
            return vals
        return None

    def _as_vector(self, param_values: dict | Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Normalise l'entrée paramètres en vecteur numpy, dans l'ordre `self.parameters`.
        - dict : {name: value}
        - iterable / np.ndarray : déjà ordonné (même ordre que self.parameters)
        """
        if isinstance(param_values, dict):
            vec = np.array([param_values[p.name] for p in self.parameters], dtype=float)
        else:
            vec = np.asarray(list(param_values), dtype=float)
        if vec.size != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} parameter values, got {vec.size}."
            )
        return vec

    def _to_property_dict(self, vec: np.ndarray) -> dict[str, float]:
        """
        Construit le dict de propriétés pour OMModel.set_param_dict.
        - Si `model_property` est défini, on l’utilise (str ou tuple de str).
        - Sinon on utilise `Parameter.name`.
        Si un tuple de propriétés est donné, on affecte la même valeur scalaire à chaque propriété.
        """
        prop_dict: dict[str, float] = {}
        for p, v in zip(self.parameters, vec):
            target = p.model_property if p.model_property is not None else p.name
            if isinstance(target, tuple):
                for path in target:
                    prop_dict[str(path)] = float(v)
            else:
                prop_dict[str(target)] = float(v)
        return prop_dict

    def function(self, param_values: dict | Iterable[float] | np.ndarray, kwargs: dict | None = None) -> dict[str, float]:
        _ = {} if kwargs is None else kwargs

        vec = self._as_vector(param_values)
        property_dict = self._to_property_dict(vec)

        self.om_model.set_param_dict(property_dict)

        sim_df = self.om_model.simulate()

        if not isinstance(sim_df, (pd.DataFrame, pd.Series)):
            raise TypeError("OMModel.simulate must return a pandas DataFrame or Series.")

        sim_df = sim_df if isinstance(sim_df, pd.DataFrame) else sim_df.to_frame()

        out: dict[str, float] = {}
        for ind, spec in self.indicators_config.items():
            if ind not in sim_df.columns:
                raise KeyError(f"Indicator {ind!r} not found in simulation outputs: {list(sim_df.columns)}.")

            series = sim_df[ind]
            if isinstance(spec, tuple):
                func, ref = spec
                out[ind] = float(func(series, ref))
            else:
                func = spec
                out[ind] = float(func(series))

        return out

    def scipy_obj_function(self, x: float | Iterable[float] | np.ndarray, kwargs: dict | None = None) -> float:
        if isinstance(x, (float, int)):
            x_vec = np.array([x], dtype=float)
        else:
            x_vec = np.asarray(list(x), dtype=float)

        if x_vec.size != len(self.parameters):
            raise ValueError("Length of x does not match number of parameters.")

        res = self.function(x_vec, kwargs)
        if self.scipy_obj_indicator not in res:
            raise KeyError(
                f"scipy_obj_indicator {self.scipy_obj_indicator!r} not computed. "
                f"Available: {list(res.keys())}"
            )
        return float(res[self.scipy_obj_indicator])
