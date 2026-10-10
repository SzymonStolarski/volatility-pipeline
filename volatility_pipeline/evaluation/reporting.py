"""
Tools that operate on a whole set of ForecastResults after an evaluation:
re-scoring, sub-setting and reporting. None of them refits a model.

  rescore              the same forecasts against a different proxy — the r^2
                       robustness column of the article
  exclude_dates        the same evaluation without some days — the main run
                       scored without contract-roll days, which is what a run
                       on data with those days removed should be compared to
  hyperparameter_table the settings every ML model actually ran with, for the
                       supplement (tuned values used to stay inside the worker
                       processes and were never reported)
"""
from __future__ import annotations

import pandas as pd

from .rolling_forecast import ForecastResult


def rescore(results: dict, actuals: pd.Series, proxy: str | None = None) -> dict:
    """{name: result scored against `actuals`} with forecasts unchanged."""
    return {n: r.with_actuals(actuals, proxy) for n, r in results.items()}


def exclude_dates(results: dict, dates) -> dict:
    """{name: result without `dates`} with forecasts unchanged."""
    return {n: r.drop_dates(dates) for n, r in results.items()}


def hyperparameter_table(results: dict) -> pd.DataFrame:
    """
    One row per model that reports hyperparameters (the ML families), one
    column per setting, plus the tuning cadence and how many searches ran.
    Settings a model does not have are left blank.
    """
    rows = []
    for name, r in results.items():
        hp = (r.meta or {}).get("hyperparameters")
        if not hp:
            continue
        rows.append({"model": name, "tune": r.meta.get("tune"),
                     "n_searches": r.meta.get("n_searches"), **hp})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model")
