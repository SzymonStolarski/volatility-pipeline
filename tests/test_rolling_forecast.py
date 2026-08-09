"""
Regression tests for the rolling out-of-sample evaluation.

The defect these exist to prevent: RollingEvaluator re-estimated parameters
every `refit_every` steps but never advanced the model's state in between, so
`forecast_variance()` returned the value computed at the last re-fit. With
refit_every=10 that made 90% of a supposedly one-step-ahead evaluation stale,
and produced exactly n_test/refit_every distinct forecasts.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.evaluation.proxies import garman_klass
from volatility_pipeline.evaluation.rolling_forecast import RollingEvaluator
from volatility_pipeline.models.garch_models import GARCHModel, make_garch
from volatility_pipeline.models.xgb_models import XGBHybridModel, XGBVolatilityModel

REFIT = 10


@pytest.fixture(scope="module")
def data():
    """Synthetic GARCH(1,1) returns plus an OHLC-style proxy — no network."""
    rng = np.random.default_rng(0)
    n = 900
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
    # A less noisy proxy for the same h: chi-square with more degrees of freedom.
    proxy = pd.Series(h * rng.chisquare(df=8, size=n) / 8.0, index=idx, name="proxy")
    return returns.iloc[:700], returns.iloc[700:], proxy


# --------------------------------------------------------------------------
# Fix 1 — the forecast must move every day
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window_type,window_size", [("expanding", None), ("sliding", 400)])
def test_garch_forecasts_are_not_frozen(data, window_type, window_size):
    train, test, _ = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT,
                          window_type=window_type, window_size=window_size)
    res = ev.evaluate(lambda: make_garch("GARCH", "normal"), "g", train, test)
    f = res.forecasts.values
    assert len(np.unique(f)) == len(f), (
        f"{len(np.unique(f))} distinct forecasts for {len(f)} test days — "
        f"state is not advancing between re-fits"
    )
    assert len(res.refit_indices) == int(np.ceil(len(test) / REFIT))


def test_xgb_and_hybrid_forecasts_are_not_frozen(data):
    train, test, _ = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT)
    for factory, label in [
        (lambda: XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1), "xgb"),
        (lambda: XGBHybridModel(n_lags=5, use_optuna=False, seed=1), "hybrid"),
    ]:
        f = ev.evaluate(factory, label, train, test).forecasts.values
        assert len(np.unique(f)) == len(f), f"{label}: frozen between re-fits"


def test_forecast_on_refit_days_is_unchanged_by_the_fix(data):
    """
    On a re-fit day fit() has just consumed every observation, so the forecast
    must equal a plain fit-then-forecast. This is the invariant that pins the
    fix to *only* affecting the stale days.
    """
    train, test, _ = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT)
    res = ev.evaluate(lambda: make_garch("GARCH", "normal"), "g", train, test)
    all_returns = pd.concat([train, test])
    for i in res.refit_indices:
        m = make_garch("GARCH", "normal").fit(all_returns.iloc[: len(train) + i])
        assert res.forecasts.values[i] == pytest.approx(
            float(m.forecast_variance(1)[0]), rel=1e-10
        )


def test_update_with_the_fit_sample_is_a_no_op(data):
    """update() must hold the parameters and only re-filter the state."""
    train, _, _ = data
    m = GARCHModel("GARCH", "normal").fit(train)
    before = float(m.forecast_variance(1)[0])
    params_before = m.params.copy()
    m.update(train)
    assert float(m.forecast_variance(1)[0]) == pytest.approx(before, rel=1e-12)
    pd.testing.assert_series_equal(m.params, params_before)


def test_update_changes_the_forecast_when_data_arrives(data):
    train, test, _ = data
    m = GARCHModel("GARCH", "normal").fit(train)
    before = float(m.forecast_variance(1)[0])
    m.update(pd.concat([train, test.iloc[:5]]))
    assert float(m.forecast_variance(1)[0]) != pytest.approx(before, rel=1e-9)


def test_missing_update_method_fails_loudly(data):
    train, test, _ = data

    class Frozen:
        def fit(self, returns, target=None):
            return self

        def forecast_variance(self, horizon=1):
            return np.full(horizon, 1e-4)

    with pytest.raises(TypeError, match="no update"):
        RollingEvaluator(refit_every=REFIT).evaluate(Frozen, "frozen", train, test)


# --------------------------------------------------------------------------
# Fix 2 — ML models train on the proxy they are scored against
# --------------------------------------------------------------------------

def test_ml_target_defaults_to_squared_returns(data):
    train, _, _ = data
    a = XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1).fit(train)
    b = XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1).fit(train, target=train ** 2)
    assert float(a.forecast_variance(1)[0]) == pytest.approx(float(b.forecast_variance(1)[0]))


def test_ml_target_changes_the_fit(data):
    train, _, proxy = data
    a = XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1).fit(train)
    b = XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1).fit(
        train, target=proxy.reindex(train.index)
    )
    assert float(a.forecast_variance(1)[0]) != pytest.approx(float(b.forecast_variance(1)[0]))


def test_evaluator_passes_the_proxy_through_to_training(data):
    """A full-history proxy must reach fit(); a test-only one must not break it."""
    train, test, proxy = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT)
    seen = []

    class Spy(XGBVolatilityModel):
        def fit(self, returns, target=None):
            seen.append(None if target is None else len(target))
            return super().fit(returns, target)

    ev.evaluate(lambda: Spy(n_lags=5, use_optuna=False, seed=1), "full",
                train, test, actuals_series=proxy)
    assert all(n is not None for n in seen)

    seen.clear()
    ev.evaluate(lambda: Spy(n_lags=5, use_optuna=False, seed=1), "test-only",
                train, test, actuals_series=proxy.reindex(test.index))
    assert all(n is None for n in seen), "a test-only proxy must not be used as a target"


def test_log_target_forecasts_are_strictly_positive(data):
    """
    The failure this prevents: a level-scale booster predicting <= 0, floored at
    1e-10, which sends QLIKE (log h + proxy/h) to five or six figures.
    """
    train, test, proxy = data
    ev = RollingEvaluator(n_ahead=1, refit_every=REFIT)
    res = ev.evaluate(lambda: XGBVolatilityModel(n_lags=5, use_optuna=False, seed=1),
                      "xgb", train, test, actuals_series=proxy)
    f = res.forecasts.values
    assert (f > 0).all()
    assert (f > 1e-9).all(), "forecasts hit the numerical floor — log target not applied"
    actuals = res.actuals.values
    assert np.isfinite(np.mean(np.log(f) + actuals / f))


def test_residual_hybrid_keeps_the_level_scale():
    """Residual mode targets proxy - h, which is signed: no log transform."""
    m = XGBHybridModel(mode="residual", log_target=True)
    assert m.log_target is False
    assert XGBHybridModel(mode="features", log_target=True).log_target is True


def test_garch_ignores_the_target_argument(data):
    """GARCH is estimated by QML on returns — the proxy must not touch it."""
    train, _, proxy = data
    a = GARCHModel("GARCH", "normal").fit(train)
    b = GARCHModel("GARCH", "normal").fit(train, target=proxy.reindex(train.index))
    pd.testing.assert_series_equal(a.params, b.params)


def test_garman_klass_can_be_non_positive():
    """
    Documents real behaviour: GK is not bounded below by zero. BZ=F has 80 days
    with H == L (proxy exactly 0) and 7 where the close sits outside [L, H]
    (proxy negative), all before 2020. The docstring used to claim otherwise.
    """
    idx = pd.period_range("2020-01-01", periods=2, freq="D")
    flat = garman_klass(pd.Series([50.0, 50.0], index=idx), pd.Series([50.0, 50.2], index=idx),
                        pd.Series([50.0, 50.1], index=idx), pd.Series([50.0, 49.0], index=idx))
    assert flat.iloc[0] == 0.0
    assert flat.iloc[1] < 0.0
