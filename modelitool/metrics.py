import numpy as np


def nmbe(y_pred, y_true):
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("x size error")

    return ((y_pred - y_true).sum() / y_true.sum()) * 100


def cv_rmse(y_pred, y_true):
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("x size error")

    return (
            (1 / y_true.mean()) *
            np.sqrt(sum((y_true - y_pred) ** 2 / (y_true.shape[0] - 1))) *
            100
    )
