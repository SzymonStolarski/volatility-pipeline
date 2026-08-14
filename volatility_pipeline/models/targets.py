"""
Realized-variance target construction, shared by the ML model families.

The GARCH family is estimated by quasi-maximum likelihood on the returns
themselves and has no regression target, so nothing here applies to it. The ML
models do have one, and two properties of that target matter:

1. WHICH PROXY.  The evaluator scores every model against a realized-variance
   proxy chosen by the caller (squared returns, Parkinson, Garman-Klass). The ML
   models should learn against the same one. Squared returns and the range-based
   proxies are both unbiased for sigma^2, but r^2 is several times noisier, so
   training on r^2 while scoring on Garman-Klass throws away signal for nothing.

2. WHICH SCALE.  Fitting a level-scale target under squared error lets a tree
   ensemble predict zero or negative variance, which QLIKE (log h + proxy/h)
   punishes without bound. Fitting log-variance and retransforming makes the
   forecast positive by construction.
"""
from __future__ import annotations
import numpy as np


_EPS = 1e-12


def winsor_bounds(x: np.ndarray, limits: tuple[float, float]) -> tuple[float, float]:
    lo_q, hi_q = limits
    lo, hi = np.quantile(x, [lo_q, 1.0 - hi_q])
    return float(lo), float(hi)


def winsorize(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(x, bounds[0], bounds[1])


def log_variance_target(values: np.ndarray, lower_q: float) -> np.ndarray:
    """
    log of a realized-variance proxy, floored so that near-zero days cannot
    dominate the fit.

    A day whose proxy is 0 (r_t = 0 on a stale or unchanged close — 21 such days
    in the BZ=F training window; or a zero-range OHLC record, of which BZ=F has
    80 before 2020) gives log(0 + _EPS) = -27.6, roughly 6 standard deviations
    below the mean log target. Those points contribute almost all of the squared
    error being minimised, so the model fits the zeros instead of the volatility
    dynamics.

    Only the LOWER tail is clipped. The upper tail is left intact: capping it
    would systematically cut the forecast during exactly the high-volatility
    episodes the model exists to predict.
    """
    finite = np.asarray(values, dtype=float)
    lo = float(np.quantile(finite, lower_q))
    if not np.isfinite(lo) or lo <= 0.0:
        positive = finite[finite > 0]
        lo = float(positive.min()) if positive.size else _EPS
    return np.log(np.maximum(finite, lo))


def smearing_factor(log_actual: np.ndarray, log_pred: np.ndarray) -> float:
    """
    Duan (1983) smearing estimate, S = mean(exp(u_i)) over the fit residuals
    u_i = log y_i - log yhat_i.

    A model trained on log-variance under squared error estimates
    E[log proxy | X], so exp(.) of its output estimates the CONDITIONAL
    GEOMETRIC mean, not the conditional variance E[proxy | X] that every other
    model in the pipeline forecasts and that the loss functions score. For
    r_t = sqrt(h_t) z_t the two differ by the constant factor
    exp(E[log z^2]) = exp(-1.27) ~ 0.28 under normality, and by considerably
    more under fat tails: measured on BZ=F the raw exponentiated forecast was
    0.18x the GARCH level, which QLIKE (log h + proxy/h) punishes severely.

    Multiplying by S removes that bias without assuming log-normal residuals.
    """
    resid = np.asarray(log_actual, dtype=float) - np.asarray(log_pred, dtype=float)
    s = float(np.mean(np.exp(resid)))
    # A degenerate fit (non-finite or non-positive S) must not silently scale
    # every forecast to nonsense; fall back to the uncorrected level.
    return s if (np.isfinite(s) and s > 0.0) else 1.0


def resolve_target(returns: np.ndarray, target: np.ndarray | None) -> np.ndarray:
    """
    The realized-variance series the ML models should learn against.

    `target` is the proxy the evaluator will score against, aligned to
    `returns`. When it is None the caller did not supply one and we fall back to
    squared returns, which is what the pipeline did before the proxy was
    plumbed through to training.
    """
    r = np.asarray(returns, dtype=float)
    if target is None:
        return r ** 2
    t = np.asarray(target, dtype=float)
    if t.shape != r.shape:
        raise ValueError(
            f"target must align with returns: got {t.shape} vs {r.shape}. "
            f"Pass the proxy reindexed to the training window."
        )
    if not np.isfinite(t).all():
        raise ValueError(
            "target contains non-finite values — reindex the proxy onto the "
            "returns index before passing it in."
        )
    return t
