import numpy as np


def metric(pred, true, eps=1e-6):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    diff = pred - true
    abs_diff = np.abs(diff)

    mae = np.mean(abs_diff)
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mape = np.mean(abs_diff / np.maximum(np.abs(true), eps))
    smape = np.mean(2.0 * abs_diff / np.maximum(np.abs(pred) + np.abs(true), eps))
    wape = np.sum(abs_diff) / np.maximum(np.sum(np.abs(true)), eps)

    denom = np.sum((true - np.mean(true)) ** 2)
    nse = 1.0 - np.sum(diff**2) / np.maximum(denom, eps)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "wape": float(wape),
        "nse": float(nse),
    }
