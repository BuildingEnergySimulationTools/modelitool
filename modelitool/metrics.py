import numpy as np
import pandas as pd


def check_pd_flatten(*args):
    res = []
    for obj in args:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            res.append(obj.to_numpy().flatten())
        else:
            res.append(obj)
    return res


def nmbe(y_pred, y_true):
    y_pred, y_true = check_pd_flatten(y_pred, y_true)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("x size error")

    return (
        np.sum((y_pred - y_true)) /
        np.sum(y_true) *
        100
    )


def cv_rmse(y_pred, y_true):
    y_pred, y_true = check_pd_flatten(y_pred, y_true)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("x size error")

    return (
            (1 / np.mean(y_true)) *
            np.sqrt(
                np.sum((y_true - y_pred) ** 2) /
                (y_true.shape[0] - 1)
            ) * 100
    )
