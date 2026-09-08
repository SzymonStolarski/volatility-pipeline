"""
Tests for the realized-variance proxies.

Two defects these exist to prevent:

1. A GARCH model estimated on close-to-close returns forecasts the FULL daily
   variance, but Garman-Klass and Parkinson measure the intraday session only.
   Scoring one against the other penalises the model by whatever share of
   variance happens overnight — 37% on NG=F over this project's test window.
   The overnight-inclusive proxies exist to close that gap.

2. The overnight term needs the close from the day BEFORE the first return, and
   the returns series has already dropped that day. Reindexing the OHLC inputs
   onto the returns index before differencing silently loses one observation and
   leaves a NaN that the evaluator then rejects. `compute_proxy` exists to make
   that impossible; `test_compute_proxy_keeps_the_first_observation` pins it.

The estimators are validated against simulated intraday paths with a KNOWN
volatility, which is the only way to check a range estimator: on real data the
true variance is never observed.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.evaluation.proxies import (
    PROXY_REGISTRY,
    compute_proxy,
    garman_klass,
    garman_klass_overnight,
    overnight_gap_report,
    overnight_variance,
    parkinson,
    rogers_satchell,
    rogers_satchell_overnight,
    squared_returns,
    yang_zhang,
)

SIG_INTRADAY = 0.012
SIG_OVERNIGHT = 0.006
TRUE_INTRADAY = SIG_INTRADAY ** 2
TRUE_OVERNIGHT = SIG_OVERNIGHT ** 2
TRUE_TOTAL = TRUE_INTRADAY + TRUE_OVERNIGHT


def simulate_ohlc(n_days=2500, steps=20000, drift=0.0, seed=0):
    """
    Daily OHLC from a simulated within-day GBM path plus an overnight jump.

    `steps` must be large: the observed high/low of a discretely sampled path
    understate the continuous-time extremes, which biases every range estimator
    DOWNWARD (~20% at 78 steps/day, ~9% at 390, <1% at 20000). That is a real
    property of range estimators, not an implementation error, so the tolerance
    here is only meaningful at fine sampling.
    """
    rng = np.random.default_rng(seed)
    ds = SIG_INTRADAY / np.sqrt(steps)
    mu = drift / steps
    log_prev_close = np.log(100.0)
    O, H, L, C = (np.empty(n_days) for _ in range(4))
    for t in range(n_days):
        log_open = log_prev_close + rng.standard_normal() * SIG_OVERNIGHT
        path = log_open + np.cumsum(rng.standard_normal(steps) * ds + mu)
        O[t] = np.exp(log_open)
        H[t] = np.exp(max(path.max(), log_open))
        L[t] = np.exp(min(path.min(), log_open))
        C[t] = np.exp(path[-1])
        log_prev_close = path[-1]
    idx = pd.date_range("2010-01-01", periods=n_days, freq="B")
    s = lambda a: pd.Series(a, index=idx)
    return s(O), s(H), s(L), s(C)


@pytest.fixture(scope="module")
def ohlc():
    return simulate_ohlc()


@pytest.fixture(scope="module")
def ohlc_drifting():
    """Strong upward drift: one full intraday sigma per day."""
    return simulate_ohlc(drift=0.012, seed=1)


# --------------------------------------------------------------------------
# The estimators recover a known variance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["garman_klass", "parkinson", "rogers_satchell"])
def test_intraday_estimators_recover_true_intraday_variance(ohlc, name):
    O, H, L, C = ohlc
    est = parkinson(H, L) if name == "parkinson" else {
        "garman_klass": garman_klass, "rogers_satchell": rogers_satchell
    }[name](O, H, L, C)
    assert abs(float(est.mean()) / TRUE_INTRADAY - 1.0) < 0.05


def test_intraday_estimators_exclude_the_overnight_move(ohlc):
    """They must land on the intraday variance, NOT the total — this gap is the
    whole reason the overnight-inclusive proxies exist."""
    O, H, L, C = ohlc
    assert float(garman_klass(O, H, L, C).mean()) < 0.8 * TRUE_TOTAL


def test_overnight_variance_recovers_the_overnight_component(ohlc):
    O, H, L, C = ohlc
    on = overnight_variance(O, C)
    assert np.isnan(on.iloc[0]), "first day has no preceding close"
    assert on.iloc[1:].notna().all()
    assert abs(float(on.mean()) / TRUE_OVERNIGHT - 1.0) < 0.10


@pytest.mark.parametrize("func", [garman_klass_overnight, rogers_satchell_overnight])
def test_full_variance_proxies_recover_the_total(ohlc, func):
    O, H, L, C = ohlc
    assert abs(float(func(O, H, L, C).mean()) / TRUE_TOTAL - 1.0) < 0.05


def test_full_variance_proxies_are_exactly_intraday_plus_overnight(ohlc):
    """No k-weighting is applied per-day: the components must add up exactly."""
    O, H, L, C = ohlc
    on = overnight_variance(O, C)
    pd.testing.assert_series_equal(
        garman_klass_overnight(O, H, L, C),
        (garman_klass(O, H, L, C) + on).rename("garman_klass_overnight"),
    )
    pd.testing.assert_series_equal(
        rogers_satchell_overnight(O, H, L, C),
        (rogers_satchell(O, H, L, C) + on).rename("rogers_satchell_overnight"),
    )


# --------------------------------------------------------------------------
# Drift independence — the reason Rogers-Satchell is offered alongside GK
# --------------------------------------------------------------------------

def test_rogers_satchell_is_drift_independent(ohlc_drifting):
    O, H, L, C = ohlc_drifting
    assert abs(float(rogers_satchell(O, H, L, C).mean()) / TRUE_INTRADAY - 1.0) < 0.05


def test_parkinson_is_inflated_by_drift(ohlc_drifting):
    """Contrast case: GK and Parkinson attribute drift-driven range to
    volatility, so they must NOT survive the same check."""
    O, H, L, C = ohlc_drifting
    assert float(parkinson(H, L).mean()) / TRUE_INTRADAY > 1.15


# --------------------------------------------------------------------------
# Yang-Zhang
# --------------------------------------------------------------------------

def test_yang_zhang_recovers_the_total_variance(ohlc):
    O, H, L, C = ohlc
    yz = yang_zhang(O, H, L, C, window=20)
    assert abs(float(yz.mean()) / TRUE_TOTAL - 1.0) < 0.05


def test_yang_zhang_warmup_is_nan(ohlc):
    O, H, L, C = ohlc
    yz = yang_zhang(O, H, L, C, window=20)
    assert yz.iloc[:20].isna().all()
    assert yz.iloc[20:].notna().all()


def test_yang_zhang_rejects_a_single_day_window(ohlc):
    O, H, L, C = ohlc
    with pytest.raises(ValueError, match="window >= 2"):
        yang_zhang(O, H, L, C, window=1)


def test_yang_zhang_is_smoother_than_a_per_day_proxy(ohlc):
    """
    The reason it must not be used to score one-step-ahead forecasts:
    consecutive values share window-1 inputs, so it is heavily autocorrelated
    and far smoother than the quantity being forecast.
    """
    O, H, L, C = ohlc
    yz = yang_zhang(O, H, L, C, window=20).dropna()
    per_day = rogers_satchell_overnight(O, H, L, C).dropna()
    assert yz.autocorr(1) > 0.85
    assert per_day.autocorr(1) < 0.15


def test_yang_zhang_is_not_a_scoring_proxy():
    assert "yang_zhang" not in PROXY_REGISTRY
    with pytest.raises(ValueError, match="multi-day estimator"):
        compute_proxy("yang_zhang", returns=pd.Series([0.01, -0.01]))


# --------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------

def test_compute_proxy_keeps_the_first_observation(ohlc):
    """
    The off-by-one this function exists to prevent. `returns` drops the first
    day, so the overnight term for the FIRST return needs the close from a day
    that is no longer in returns.index. Reindexing the inputs first would make
    that value NaN.
    """
    O, H, L, C = ohlc
    returns = np.log(C / C.shift(1)).dropna()
    proxy = compute_proxy("garman_klass_overnight",
                          returns=returns, open_=O, high=H, low=L, close=C)
    assert len(proxy) == len(returns)
    assert proxy.notna().all()
    assert proxy.index.equals(returns.index)


def test_compute_proxy_rejects_pre_sliced_ohlc(ohlc):
    """Passing OHLC already cut to returns.index must fail loudly, not silently
    return a NaN in the first slot."""
    O, H, L, C = ohlc
    returns = np.log(C / C.shift(1)).dropna()
    with pytest.raises(ValueError, match="non-finite"):
        compute_proxy("garman_klass_overnight", returns=returns,
                      open_=O.reindex(returns.index), high=H.reindex(returns.index),
                      low=L.reindex(returns.index), close=C.reindex(returns.index))


@pytest.mark.parametrize("name", sorted(PROXY_REGISTRY))
def test_compute_proxy_dispatches_every_registered_proxy(ohlc, name):
    O, H, L, C = ohlc
    returns = np.log(C / C.shift(1)).dropna()
    proxy = compute_proxy(name, returns=returns, open_=O, high=H, low=L, close=C)
    assert proxy.index.equals(returns.index)
    assert proxy.notna().all()
    assert proxy.name == name


def test_compute_proxy_needs_ohlc_for_range_proxies(ohlc):
    O, H, L, C = ohlc
    returns = np.log(C / C.shift(1)).dropna()
    with pytest.raises(ValueError, match="needs OHLC"):
        compute_proxy("garman_klass", returns=returns)


def test_compute_proxy_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown variance proxy"):
        compute_proxy("realized_kernel", returns=pd.Series([0.01, -0.01]))


def test_squared_returns_needs_no_ohlc():
    r = pd.Series([0.01, -0.02, 0.005])
    pd.testing.assert_series_equal(
        compute_proxy("squared", returns=r), (r ** 2).rename("squared")
    )


# --------------------------------------------------------------------------
# Contract-roll diagnostic
# --------------------------------------------------------------------------

def test_overnight_gap_report_finds_a_planted_roll(ohlc):
    O, H, L, C = ohlc
    original_gap = np.log(O.iloc[500] / C.iloc[499])
    O = O.copy()
    O.iloc[500] = O.iloc[500] * 0.5     # halve the open: a contract-roll-sized gap

    rep = overnight_gap_report(O, C, threshold=0.15)
    assert rep["n_flagged"] == 1
    assert rep["flagged"].index[0] == O.index[500]
    # one fake day must dominate the whole overnight sum
    assert rep["share_of_overnight_sum"] > 0.5
    assert rep["mean_without"] < rep["mean_with"]
    # halving the open shifts that day's gap by log(0.5) on top of the real jump
    assert float(rep["flagged"]["overnight_return"].iloc[0]) == pytest.approx(
        np.log(0.5) + original_gap
    )


def test_overnight_gap_report_is_quiet_on_clean_data(ohlc):
    O, H, L, C = ohlc
    rep = overnight_gap_report(O, C, threshold=0.15)
    assert rep["n_flagged"] == 0
    assert rep["share_of_overnight_sum"] == 0.0
    assert rep["mean_without"] == pytest.approx(rep["mean_with"])
