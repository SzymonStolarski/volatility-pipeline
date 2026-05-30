from __future__ import annotations
import numpy as np


def rmse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(forecasts) - np.asarray(actuals)) ** 2)))


def mae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(forecasts) - np.asarray(actuals))))


def mse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((np.asarray(forecasts) - np.asarray(actuals)) ** 2))


def qlike(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Quasi-likelihood loss: mean(log(h_t) + r_t^2 / h_t)."""
    h = np.asarray(forecasts, dtype=float)
    r2 = np.asarray(actuals, dtype=float)
    return float(np.mean(np.log(h) + r2 / h))


# Per-period loss functions returning arrays (used by DM test and MCS)
LOSS_FUNCTIONS: dict = {
    "squared":  lambda f, a: (np.asarray(f) - np.asarray(a)) ** 2,
    "absolute": lambda f, a: np.abs(np.asarray(f) - np.asarray(a)),
    "qlike":    lambda f, a: np.log(np.asarray(f)) + np.asarray(a) / np.asarray(f),
}


def compute_loss(forecasts: np.ndarray, actuals: np.ndarray, loss: str = "squared") -> np.ndarray:
    if loss not in LOSS_FUNCTIONS:
        raise ValueError(f"loss must be one of {list(LOSS_FUNCTIONS)}, got {loss!r}")
    return LOSS_FUNCTIONS[loss](forecasts, actuals)


def metrics_summary(forecasts: np.ndarray, actuals: np.ndarray) -> dict:
    return {
        "RMSE":  rmse(forecasts, actuals),
        "MAE":   mae(forecasts, actuals),
        "MSE":   mse(forecasts, actuals),
        "QLIKE": qlike(forecasts, actuals),
    }
