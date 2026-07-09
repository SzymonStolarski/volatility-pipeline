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

    Parameters
    ----------
    results   : dict mapping model name -> ForecastResult (anything with
                `.forecasts` and `.actuals` pd.Series)
    ratio     : how many times outside the realized-proxy range a forecast may
                fall before it is considered extreme
    qlike_cap : |QLIKE| above this is treated as an explosion
    warn      : emit a RuntimeWarning per degenerate model (default True)

    Returns
    -------
    DataFrame indexed by model name with the per-model screen and a boolean
    `degenerate` column, sorted so degenerate models appear first.
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

        rows[name] = {
            "n": int(f.size),
            "min_forecast": float(np.nanmin(f)) if f.size else float("nan"),
            "max_forecast": float(np.nanmax(f)) if f.size else float("nan"),
            "n_nonfinite": n_nonfinite,
            "n_nonpositive": n_nonpositive,
            "n_extreme_low": n_low,
            "n_extreme_high": n_high,
            "RMSE": rm,
            "QLIKE": ql,
            "degenerate": degenerate,
        }

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
    # Preserve declared dtypes and sort degenerate models to the top.
    df["degenerate"] = df["degenerate"].astype(bool)
    for col in ("n", "n_nonfinite", "n_nonpositive", "n_extreme_low", "n_extreme_high"):
        df[col] = df[col].astype(int)
    return df.sort_values(["degenerate", "QLIKE"], ascending=[False, False])
