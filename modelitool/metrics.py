import numpy as np
import pandas as pd


def check_shape(ym, yt):
    if ym.shape != yt.shape:
        raise ValueError(
            f"y_pred shape {ym.shape} doesn't match y_true shape {yt.shape}"
        )


def check_pd_flatten(*args):
    res = []
    for obj in args:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            res.append(obj.to_numpy().flatten())
        else:
            res.append(obj)
    return res


def nmbe(y_pred, y_true):
    check_shape(y_pred, y_true)
    y_pred, y_true = check_pd_flatten(y_pred, y_true)

    return np.sum((y_pred - y_true)) / np.sum(y_true) * 100


def cv_rmse(y_pred, y_true):
    check_shape(y_pred, y_true)
    y_pred, y_true = check_pd_flatten(y_pred, y_true)

    return (
        (1 / np.mean(y_true))
        * np.sqrt(np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] - 1))
        * 100
    )
