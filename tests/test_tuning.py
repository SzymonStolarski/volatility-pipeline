"""
Tests for hyperparameter-tuning cadence.

The defect these exist to prevent is an asymmetric comparison. XGBoost used to
run 50 Optuna trials inside every fit() — ~101 searches over a rolling
evaluation — while the LSTM was never tuned at all. Comparing the two, or the
hybrids built on them, then measures tuning effort as much as model class.

The load-bearing tests are `test_cadence_through_the_evaluator` and its LSTM
counterpart: they count ACTUAL searches across a full rolling evaluation, which
is the only place the distinction between "first" and "always" is observable.
RollingEvaluator builds a fresh model at every re-estimation, so "tune once"
depends on a cache outliving the model instance, and a mistake there fails
silently by reverting to the old behaviour.
"""
import numpy as np
import pandas as pd
import pytest

import volatility_pipeline.models.lstm_models as lstm_mod
import volatility_pipeline.models.xgb_models as xgb_mod
from volatility_pipeline.evaluation.rolling_forecast import RollingEvaluator
from volatility_pipeline.models import (
    LSTMHybridModel,
    LSTMVolatilityModel,
    XGBHybridModel,
    XGBVolatilityModel,
)
from volatility_pipeline.models.lstm_models import LSTM_SEARCH_SPACE, LSTM_TUNABLE
from volatility_pipeline.models.tuning import (
    TuningCache,
    resolve_hyperparameters,
    tuned_factory,
    validate_tune,
)

REFIT = 20


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 700
    omega, alpha, beta = 2.0e-6, 0.08, 0.90
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    idx = pd.period_range("2015-01-01", periods=n, freq="D")
    returns = pd.Series(r, index=idx)
    proxy = pd.Series(h * rng.chisquare(8, n) / 8.0, index=idx, name="gk")
    return returns.iloc[:600], returns.iloc[600:], proxy


# --------------------------------------------------------------------------
# The cache and the resolver
# --------------------------------------------------------------------------

def test_validate_tune_rejects_unknown_modes():
    for mode in ("never", "first", "always"):
        assert validate_tune(mode) == mode
    with pytest.raises(ValueError, match="tune must be one of"):
        validate_tune("sometimes")


def test_resolver_never_runs_no_search():
    calls = []
    out = resolve_hyperparameters(
        "never", TuningCache(), lambda: calls.append(1) or {"a": 2}, {"a": 1}
    )
    assert out == {"a": 1} and calls == []


def test_resolver_always_searches_every_time():
    cache = TuningCache()
    calls = []
    fn = lambda: (calls.append(1), {"a": len(calls)})[1]
    for _ in range(3):
        resolve_hyperparameters("always", cache, fn, {"a": 0})
    assert len(calls) == 3
    assert cache.n_searches == 3


def test_resolver_first_searches_once_then_reuses():
    cache = TuningCache()
    calls = []
    fn = lambda: (calls.append(1), {"a": len(calls)})[1]
    results = [resolve_hyperparameters("first", cache, fn, {"a": 0}) for _ in range(5)]
    assert len(calls) == 1
    assert cache.n_searches == 1
    assert all(r == {"a": 1} for r in results)


def test_resolver_first_without_a_cache_raises():
    """
    The silent-degradation guard. Without a cache, "first" would search on every
    fit and be indistinguishable from "always" from the outside — which is the
    exact asymmetry this module exists to remove.
    """
    with pytest.raises(ValueError, match="needs a TuningCache"):
        resolve_hyperparameters("first", None, lambda: {"a": 1}, {"a": 0})


def test_resolver_returns_copies_not_the_cached_dict():
    cache = TuningCache()
    a = resolve_hyperparameters("first", cache, lambda: {"a": 1}, {})
    a["a"] = 999
    b = resolve_hyperparameters("first", cache, lambda: {"a": 1}, {})
    assert b["a"] == 1


def test_tuned_factory_shares_one_cache_and_is_picklable():
    import pickle
    f = tuned_factory(XGBVolatilityModel, tune="first", n_lags=3)
    m1, m2 = f(), f()
    assert m1.tuning_cache is m2.tuning_cache
    assert pickle.loads(pickle.dumps(f))().tune == "first"
    # a separate factory must not share with the first
    assert tuned_factory(XGBVolatilityModel, tune="first", n_lags=3)().tuning_cache \
        is not m1.tuning_cache


# --------------------------------------------------------------------------
# Cadence measured across a real rolling evaluation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [XGBVolatilityModel, XGBHybridModel])
@pytest.mark.parametrize("mode,expected", [("never", 0), ("first", 1), ("always", 5)])
def test_cadence_through_the_evaluator(data, monkeypatch, model_cls, mode, expected):
    """
    100 test days at refit_every=20 is 5 re-estimations, so 'always' must search
    5 times and 'first' exactly once. This is the only place the two differ.
    """
    train, test, proxy = data
    assert len(test) // REFIT == expected or mode != "always"

    calls = []
    real = xgb_mod._optuna_tune
    monkeypatch.setattr(
        xgb_mod, "_optuna_tune",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    factory = tuned_factory(model_cls, tune=mode, n_lags=3, n_trials=2, seed=1)
    RollingEvaluator(n_ahead=1, refit_every=REFIT).evaluate(
        factory, "m", train, test, actuals_series=proxy
    )
    assert len(calls) == expected


def test_first_survives_the_parallel_path(data):
    train, test, proxy = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT)
    kw = dict(tune="first", n_lags=3, n_trials=2, seed=1)
    seq = ev.evaluate(tuned_factory(XGBVolatilityModel, **kw), "m",
                      train, test, actuals_series=proxy)
    par = ev.evaluate_many([(tuned_factory(XGBVolatilityModel, **kw), "m")],
                           train, test, actuals_series=proxy,
                           verbose=False, n_jobs=2)["m"]
    pd.testing.assert_series_equal(seq.forecasts, par.forecasts)


# --------------------------------------------------------------------------
# LSTM tuning
# --------------------------------------------------------------------------

LSTM_KW = dict(n_trials=2, seed=1, max_epochs=8, patience=3, device="cpu")


def test_lstm_search_space_is_complete():
    assert set(LSTM_TUNABLE) == set(LSTM_SEARCH_SPACE)
    # everything searched must be a real constructor argument
    m = LSTMVolatilityModel()
    for name in LSTM_TUNABLE:
        assert hasattr(m, name)


def test_lstm_tuning_picks_values_from_the_space(data):
    train, _, _ = data
    cache = TuningCache()
    m = LSTMVolatilityModel(tune="first", tuning_cache=cache, **LSTM_KW).fit(train)
    assert cache.n_searches == 1
    assert m.lookback in LSTM_SEARCH_SPACE["lookback"]
    assert m.hidden_size in LSTM_SEARCH_SPACE["hidden_size"]
    assert m.num_layers in LSTM_SEARCH_SPACE["num_layers"]
    assert 0.0 <= m.dropout <= 0.4
    assert 1e-4 <= m.lr <= 1e-2
    assert m.batch_size in LSTM_SEARCH_SPACE["batch_size"]


def test_lstm_tuning_is_applied_to_the_instance(data):
    """The resolved values must be written back, since update() and the forecast
    window read self.lookback rather than anything the search returns."""
    train, _, _ = data
    cache = TuningCache()
    cache.record({"lookback": 5, "hidden_size": 16, "num_layers": 1,
                  "dropout": 0.1, "lr": 1e-3, "batch_size": 32})
    m = LSTMVolatilityModel(tune="first", tuning_cache=cache, lookback=40, **LSTM_KW)
    m.fit(train)
    assert m.lookback == 5, "cached lookback not applied"
    assert m._last_window.shape[0] == 5, "forecast window still uses the old lookback"


def test_lstm_never_leaves_constructor_values_untouched(data):
    train, _, _ = data
    m = LSTMVolatilityModel(tune="never", lookback=20, hidden_size=32, **LSTM_KW).fit(train)
    assert (m.lookback, m.hidden_size) == (20, 32)


def test_lstm_second_fit_reuses_the_cache(data):
    train, _, _ = data
    cache = TuningCache()
    a = LSTMVolatilityModel(tune="first", tuning_cache=cache, **LSTM_KW).fit(train)
    b = LSTMVolatilityModel(tune="first", tuning_cache=cache, **LSTM_KW).fit(train)
    assert cache.n_searches == 1
    assert (a.lookback, a.hidden_size, a.batch_size) == \
           (b.lookback, b.hidden_size, b.batch_size)


def test_lstm_always_warns_about_cost(data):
    """
    A full network search at every re-estimation is hours per model, not
    minutes — the user should be told rather than discover it by waiting.
    """
    train, _, _ = data
    with pytest.warns(RuntimeWarning, match="EVERY re-estimation"):
        LSTMVolatilityModel(tune="always", **LSTM_KW).fit(train)


@pytest.mark.parametrize("mode", ["features", "residual"])
def test_lstm_hybrid_tunes_in_both_modes(data, mode):
    """The hybrids inherit the asymmetry, so they must inherit the fix."""
    train, _, _ = data
    cache = TuningCache()
    m = LSTMHybridModel(mode=mode, tune="first", tuning_cache=cache, **LSTM_KW).fit(train)
    assert cache.n_searches == 1
    assert m.lookback in LSTM_SEARCH_SPACE["lookback"]
    fc = m.forecast_variance(1)[0]
    assert np.isfinite(fc) and fc > 0


def test_lstm_cadence_through_the_evaluator(data, monkeypatch):
    train, test, proxy = data
    calls = []
    real = lstm_mod._optuna_tune_lstm
    monkeypatch.setattr(
        lstm_mod, "_optuna_tune_lstm",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    factory = tuned_factory(LSTMVolatilityModel, tune="first", **LSTM_KW)
    RollingEvaluator(n_ahead=1, refit_every=REFIT).evaluate(
        factory, "lstm", train, test, actuals_series=proxy
    )
    assert len(calls) == 1, f"expected one search over {len(test) // REFIT} re-estimations"
