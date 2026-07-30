from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .metrics import rmse, qlike


def forecast_diagnostics(
    results: dict,
    *,
    ratio: float = 1e4,
    qlike_cap: float = 1e3,
    bias_tol: float = 2.0,
    warn: bool = True,
) -> pd.DataFrame:
    """
    Screen out-of-sample forecasts for numerical degeneracy that corrupts the
    loss metrics, and warn about any affected model.

    A conditional-variance forecast that collapses toward zero blows up QLIKE
    (via proxy / h_t), while one that explodes upward blows up RMSE / MSE. Either
    way a single degenerate model poisons the pairwise DM matrix and the MCS, so
    it must be caught before those tests are trusted. This was observed with
    EGARCH on short (~500-obs) sliding windows, whose forecasts oscillated
    between ~1e-25 and huge, producing QLIKE values around 1e25.

    A forecast h_t is flagged as degenerate when it is non-finite, non-positive,
    or lies more than `ratio` times outside the realized-variance-proxy range for
    that model. A model is additionally flagged when its QLIKE is non-finite or
    exceeds `qlike_cap` in absolute value (a healthy daily QLIKE is a small
    number, so |QLIKE| > 1e3 signals an explosion).

    Separately from degeneracy, the screen reports `bias_ratio` =
    mean(forecast) / mean(proxy) and flags a model as `biased` when that ratio
    falls outside [1/bias_tol, bias_tol]. Every model here forecasts the same
    quantity — the conditional variance E[r^2] — so a well-scaled forecast has a
    ratio near 1 whichever proxy is in use. A model can be perfectly smooth and
    finite and still forecast at a systematically wrong LEVEL: an LSTM trained on
    log(r^2) whose output was exponentiated without a retransformation correction
    ran at ~0.2x the realized level, passed the degeneracy screen cleanly, and
    was then eliminated first by the MCS. `biased` is deliberately NOT folded
    into `degenerate`: such a model does not corrupt the other models' DM/MCS
    results, it is simply mis-specified, and only its own numbers are suspect.

    Parameters
    ----------
    results   : dict mapping model name -> ForecastResult (anything with
                `.forecasts` and `.actuals` pd.Series)
    ratio     : how many times outside the realized-proxy range a forecast may
                fall before it is considered extreme
    qlike_cap : |QLIKE| above this is treated as an explosion
    bias_tol  : multiplicative tolerance on mean(forecast) / mean(proxy)
    warn      : emit a RuntimeWarning per degenerate or biased model (default True)

    Returns
    -------
    DataFrame indexed by model name with the per-model screen and boolean
    `degenerate` / `biased` columns, sorted so degenerate models appear first.
    """
    rows: dict[str, dict] = {}

    for name, r in results.items():
        f = np.asarray(r.forecasts.values, dtype=float)
        a = np.asarray(r.actuals.values, dtype=float)

        # Realized-proxy scale (positive values only; some range proxies can be
        # slightly negative, which we ignore for the scale reference).
        pos_a = a[np.isfinite(a) & (a > 0)]
        lo = float(pos_a.min()) if pos_a.size else 0.0
        hi = float(a[np.isfinite(a)].max()) if np.isfinite(a).any() else 0.0

        finite = np.isfinite(f)
        n_nonfinite = int((~finite).sum())
        n_nonpositive = int(np.sum(finite & (f <= 0)))

        with np.errstate(invalid="ignore"):
            n_low = int(np.sum(finite & (f > 0) & (f < lo / ratio))) if lo > 0 else 0
            n_high = int(np.sum(finite & (f > hi * ratio))) if hi > 0 else 0

        try:
            ql = float(qlike(f, a))
        except Exception:
            ql = float("nan")
        rm = float(rmse(f, a))

        qlike_bad = (not np.isfinite(ql)) or (abs(ql) > qlike_cap)
        n_bad = n_nonfinite + n_nonpositive + n_low + n_high
        degenerate = bool(n_bad or qlike_bad)

        # Systematic level bias: is the model forecasting the right magnitude?
        mean_f = float(np.mean(f[finite])) if finite.any() else float("nan")
        finite_a = np.isfinite(a)
        mean_a = float(np.mean(a[finite_a])) if finite_a.any() else float("nan")
        bias_ratio = (
            mean_f / mean_a
            if (np.isfinite(mean_f) and np.isfinite(mean_a) and mean_a > 0)
            else float("nan")
        )
        biased = bool(
            np.isfinite(bias_ratio)
            and (bias_ratio > bias_tol or bias_ratio < 1.0 / bias_tol)
        )

        rows[name] = {
            "n": int(f.size),
            "min_forecast": float(np.nanmin(f)) if f.size else float("nan"),
            "max_forecast": float(np.nanmax(f)) if f.size else float("nan"),
            "n_nonfinite": n_nonfinite,
            "n_nonpositive": n_nonpositive,
            "n_extreme_low": n_low,
            "n_extreme_high": n_high,
            "mean_forecast": mean_f,
            "mean_proxy": mean_a,
            "bias_ratio": bias_ratio,
            "RMSE": rm,
            "QLIKE": ql,
            "degenerate": degenerate,
            "biased": biased,
        }

        if warn and biased and not degenerate:
            warnings.warn(
                f"[{name}] forecasts are systematically mis-scaled — mean forecast "
                f"{mean_f:.3e} vs mean realized proxy {mean_a:.3e} "
                f"(ratio {bias_ratio:.2f}, tolerance {1/bias_tol:.2f}-{bias_tol:.2f}). "
                f"The forecasts are numerically fine, so this is a model/transform "
                f"problem, not a fitting failure — check any log/level "
                f"retransformation. QLIKE penalises this heavily and the model will "
                f"be eliminated early by the MCS.",
                RuntimeWarning,
                stacklevel=2,
            )

        if warn and degenerate:
            warnings.warn(
                f"[{name}] degenerate forecasts detected — metrics are unreliable "
                f"and this model will corrupt the DM matrix / MCS. "
                f"{n_bad} point(s) outside a sane range "
                f"(forecast h in [{np.nanmin(f):.2e}, {np.nanmax(f):.2e}]; "
                f"realized proxy ~[{lo:.2e}, {hi:.2e}]), QLIKE={ql:.3e}. "
                f"Likely a numerically unstable fit (e.g. EGARCH on a short "
                f"window). Exclude or re-fit before trusting the tests.",
                RuntimeWarning,
                stacklevel=2,
            )

    df = pd.DataFrame(rows).T
    # Preserve declared dtypes and sort problem models to the top.
    for col in ("degenerate", "biased"):
        df[col] = df[col].astype(bool)
    for col in ("n", "n_nonfinite", "n_nonpositive", "n_extreme_low", "n_extreme_high"):
        df[col] = df[col].astype(int)
    return df.sort_values(
        ["degenerate", "biased", "QLIKE"], ascending=[False, False, False]
    )
