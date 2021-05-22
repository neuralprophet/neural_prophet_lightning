from sklearn.metrics import mean_absolute_error as mae
import numpy as np


def smape(y_true, y_pred):
    return 200 * np.mean(np.abs((y_true - y_pred)) / (np.abs(y_true + y_pred)))


def mase(y_true, y_pred, y_pred_naive):
    naive_mae = mae(y_true, y_pred_naive)
    y_pred_mae = mae(y_true, y_pred)

    return y_pred_mae / naive_mae


def mues(y_true, y_pred):
    return np.mean(np.maximum(np.sign(y_true - y_pred), 0))


def moes(y_true, y_pred):
    return np.mean(np.maximum(np.sign(y_pred - y_true), 0))


def muas(y_true, y_pred):
    m = mues(y_true, y_pred)
    if m == 0:
        return 0
    else:
        return (1 / m) * np.mean(np.maximum(y_true - y_pred, 0) / np.abs(y_true))


def moas(y_true, y_pred):
    m = moes(y_true, y_pred)
    if m == 0:
        return 0
    else:
        return (1 / m) * np.mean(np.maximum(y_pred - y_true, 0) / np.abs(y_true))
