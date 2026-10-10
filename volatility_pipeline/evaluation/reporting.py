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
  equal_weight_combination
                       the equal-weight mean of several models' forecasts — the
                       GARCH-EW benchmark the combiners are measured against
"""
from __future__ import annotations

import numpy as np
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


def equal_weight_combination(results: dict, members, name: str = "GARCH-EW") -> ForecastResult:
    """
    Equal-weight combination: the arithmetic mean of the members' one-step
    variance forecasts, scored against the same actuals.

    With the GARCH family as members this is the natural benchmark for the
    combiner (Bates & Granger 1969; Timmermann 2006): the simple average of the
    same forecasts the combiner receives, which estimated combinations often
    fail to beat. The question the combiner answers is whether the ML model
    extracts from the family anything the average does not.

    Members must have been evaluated on the same dates against the same actuals.
    """
    members = list(members)
    missing = [m for m in members if m not in results]
    if missing:
        raise KeyError(f"members not among the results: {missing}")
    if len(members) < 2:
        raise ValueError("an equal-weight combination needs at least two members.")
    first = results[members[0]]
    for m in members[1:]:
        r = results[m]
        if not r.forecasts.index.equals(first.forecasts.index):
            raise ValueError(f"{m!r} was evaluated on different dates than {members[0]!r}.")
        if not np.allclose(r.actuals.to_numpy(dtype=float), first.actuals.to_numpy(dtype=float),
                           rtol=0.0, atol=0.0, equal_nan=True):
            raise ValueError(f"{m!r} is scored against different actuals than {members[0]!r}.")
    forecasts = pd.concat([results[m].forecasts for m in members], axis=1).mean(axis=1)
    return ForecastResult(
        name=name,
        forecasts=forecasts,
        actuals=first.actuals.copy(),
        refit_indices=list(first.refit_indices),
        proxy=first.proxy,
        train_target=first.train_target,
        meta={"members": members, "combination": "equal-weight mean of variance forecasts",
              "n_refits": first.meta.get("n_refits")},
    )
