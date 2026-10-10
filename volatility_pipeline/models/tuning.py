"""
Hyperparameter-tuning cadence, shared by every ML model family.

THE PROBLEM THIS SOLVES
-----------------------
XGBoost ran 50 Optuna trials inside every call to fit(). With re-estimation
every 10 steps over a 1004-day test period that is 101 searches, roughly 5,050
model fits, per XGB model per configuration. The LSTM, meanwhile, was never
tuned at all: one hard-coded value for every hyperparameter in every notebook
this repository has ever run.

So one ML family was continuously re-tuned and the other never was, which makes
any XGBoost-versus-LSTM comparison asymmetric by construction, and the same
asymmetry runs straight through the hybrids built on them. Tuning BOTH once, at
the first estimation, is the symmetric choice; it also removes a second problem,
that 101 independent searches let XGBoost's hyperparameters jump
discontinuously every ten days with no continuity between blocks.

WHY A CACHE IS NEEDED AT ALL
----------------------------
RollingEvaluator calls `model_factory()` afresh at every re-estimation, so each
refit gets a brand-new model object. "Tune once and reuse" therefore cannot live
in the model instance — it needs state that outlives it. `TuningCache` is that
state, and `tuned_factory` is the sanctioned way to wire one up.

Parallelism works out: `evaluate_many` runs one PROCESS per model spec, and all
of that spec's re-estimations happen sequentially inside it, so a cache captured
in the factory closure is populated once and reused for the rest of the run. The
mutation never needs to travel back to the parent.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

TUNE_MODES = ("never", "first", "always")


def chrono_split(
    n: int,
    *,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> int:
    """
    Split point k for n time-ordered rows: rows [0, k) are the EARLIER block
    used for fitting, rows [k, n) the LATER block used for validation. Nothing
    is ever shuffled, so validation data always post-date the data a candidate
    was trained on.

    Every split in the ML tuning path goes through here, so one tested function
    answers "is the Optuna hold-out chronological?" for all of them:

      train_frac=0.8  the hyperparameter hold-outs of both the XGBoost and the
                      LSTM search: k = int(n * train_frac)
      val_frac        the LSTM's early-stopping tail inside whatever it is
                      trained on: k = n - max(1, round(n * val_frac))

    For the LSTM the two are NESTED: a trial trains on the first 80% of the
    window, its early stopping takes the last `val_frac` of those 80%, and the
    trial is scored on the final 20%, which it has neither trained nor stopped
    on.
    """
    if (train_frac is None) == (val_frac is None):
        raise ValueError("pass exactly one of train_frac or val_frac.")
    frac = train_frac if train_frac is not None else val_frac
    if not 0.0 < frac < 1.0:
        raise ValueError(f"split fraction must lie in (0, 1), got {frac!r}.")
    if train_frac is not None:
        k = int(n * train_frac)
    else:
        k = n - max(1, int(round(n * val_frac)))
    if not 0 < k < n:
        raise ValueError(f"cannot split {n} rows into two non-empty chronological blocks.")
    return k


def validate_tune(tune: str) -> str:
    if tune not in TUNE_MODES:
        raise ValueError(f"tune must be one of {list(TUNE_MODES)}, got {tune!r}.")
    return tune


class TuningCache:
    """
    Holds one hyperparameter search result across the model instances the
    evaluator creates for a single spec.

    Not keyed by anything: one cache belongs to one model spec on one dataset.
    Re-using a single cache across commodities or across specs would hand the
    second one the first one's hyperparameters.
    """

    __slots__ = ("params", "n_searches")

    def __init__(self) -> None:
        self.params: dict | None = None
        self.n_searches: int = 0

    def record(self, params: dict) -> dict:
        self.params = dict(params)
        self.n_searches += 1
        return dict(self.params)

    def __repr__(self) -> str:
        state = "empty" if self.params is None else f"{len(self.params)} params"
        return f"TuningCache({state}, n_searches={self.n_searches})"


def resolve_hyperparameters(
    tune: str,
    cache: TuningCache | None,
    search_fn: Callable[[], dict],
    defaults: dict,
) -> dict:
    """
    Return the hyperparameters to fit with, running `search_fn` only when the
    cadence calls for it.

    never  : use `defaults`, run no search.
    first  : search on the first fit that reaches this cache, reuse thereafter.
    always : search on every fit (the previous XGBoost behaviour).

    `tune="first"` REQUIRES a cache and raises without one. "First" only means
    anything relative to a sequence of fits, and the cache is what defines that
    sequence — with `cache=None` there would be nothing to remember a previous
    search in, so every fit would search and the mode would silently collapse
    into "always". That collapse is invisible from the outside and reproduces
    precisely the asymmetry this module exists to remove, so it is made an
    error rather than a caveat.
    """
    validate_tune(tune)
    if tune == "never":
        return dict(defaults)
    if tune == "always":
        params = search_fn()
        return cache.record(params) if cache is not None else dict(params)
    if cache is None:
        raise ValueError(
            "tune='first' needs a TuningCache to remember the search across "
            "fits; without one every fit would search and the mode would "
            "silently behave like 'always'. Build evaluator factories with "
            "tuning.tuned_factory(ModelClass, tune='first', ...), which shares "
            "one cache across the model instances the evaluator creates, or "
            "pass tuning_cache=TuningCache() explicitly for a single model."
        )
    if cache.params is not None:
        return dict(cache.params)
    return cache.record(search_fn())


def _construct(model_cls, cache: TuningCache | None, kwargs: dict):
    """Module-level so `tuned_factory`'s partial stays picklable."""
    return model_cls(tuning_cache=cache, **kwargs)


def tuned_factory(model_cls, *, tune: str = "first", **kwargs) -> Callable:
    """
    Build a picklable zero-argument factory whose models share one TuningCache.

        specs = [(tuned_factory(XGBVolatilityModel, tune="first", n_lags=10),
                  "XGB-Standalone")]
        evaluator.evaluate_many(specs, ...)

    This is the only construction that makes `tune="first"` mean what it says
    under RollingEvaluator, which builds a new model at every re-estimation.
    A `lambda` would not survive pickling to a worker process; the partial does.

    One cache per call, so every spec in a list gets its own.
    """
    validate_tune(tune)
    cache = TuningCache() if tune == "first" else None
    return partial(_construct, model_cls, cache, {**kwargs, "tune": tune})
