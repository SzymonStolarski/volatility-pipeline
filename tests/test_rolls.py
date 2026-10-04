"""
Tests for contract-roll handling and the proxy floor.

The defect these exist to prevent: on a front-month futures series the
contract-roll spread sits in every quantity that compares a price from day t
with a price from day t-1 — the overnight gap AND the close-to-close return.
Cleaning only the proxy leaves the models learning from a jump the evaluation
never sees (on NG=F: a 0.73 proxy/r^2 level mismatch and a weeks-long GARCH
variance spike after every roll). The sharpest check is
`test_cleaned_series_do_not_depend_on_the_spread`: after `adjust` or `drop`,
returns and proxies must be identical whatever size the planted contract
spreads are, while the untreated series must not be.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.data import (
    calendar_roll_days,
    prepare_series,
    roll_adjusted_returns,
    roll_day_report,
)
from volatility_pipeline.evaluation.proxies import (
    compute_proxy,
    floor_proxy,
    garman_klass,
    garman_klass_overnight,
    overnight_variance,
)
from volatility_pipeline.models import LSTMHybridModel, LSTMVolatilityModel
from volatility_pipeline.models.targets import log_variance_target

TRAIN_END = "2018-12-31"


def make_ohlc(index, roll_days, spread_scale=0.2, seed=0):
    """
    OHLC for one continuous price path, quoted through a chain of contracts.

    From each roll day onward the series quotes a contract priced exp(s_k)
    relative to the previous one, so the shift appears at that day's OPEN —
    exactly how an unadjusted front-month chain behaves. Every random draw is
    made before the spreads are scaled, so two calls that differ only in
    `spread_scale` describe the same market seen through different spreads.
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    on = rng.normal(0.0, 0.005, n)
    intra = rng.normal(0.0, 0.015, n)
    up = np.abs(rng.normal(0.0, 0.006, n))
    dn = np.abs(rng.normal(0.0, 0.006, n))
    is_roll = np.asarray(index.isin(roll_days))
    spreads = rng.normal(0.0, 1.0, n) * spread_scale

    log_o, log_c = np.empty(n), np.empty(n)
    level = np.log(3.0)
    for i in range(n):
        log_o[i] = level + on[i]
        log_c[i] = log_o[i] + intra[i]
        level = log_c[i]
    shift = np.cumsum(np.where(is_roll, spreads, 0.0))

    o = np.exp(log_o + shift)
    c = np.exp(log_c + shift)
    h = np.maximum(o, c) * np.exp(up)
    l = np.minimum(o, c) * np.exp(-dn)
    mk = lambda v: pd.Series(v, index=index)
    return mk(o), mk(h), mk(l), mk(c)


@pytest.fixture(scope="module")
def calendar():
    return pd.bdate_range("2015-01-01", "2020-12-31")


@pytest.fixture(scope="module")
def ng_rolls(calendar):
    return calendar_roll_days(calendar, "NG=F")


# ---------------------------------------------------------------------------
# The calendar rules
# ---------------------------------------------------------------------------

def test_ng_rule_reproduces_the_observed_roll_dates():
    # The seven NG=F overnight gaps above 15% that are contract rolls, and the
    # one that is not (2026-02-02, a genuine Monday gap).
    idx = pd.bdate_range("2019-12-01", "2026-03-31")
    days = calendar_roll_days(idx, "NG=F")
    for d in ["2020-09-29", "2022-01-28", "2024-01-30", "2024-04-29",
              "2024-10-30", "2025-12-30", "2026-01-29"]:
        assert pd.Timestamp(d) in days, d
    assert pd.Timestamp("2026-02-02") not in days
    # exactly one roll per delivery month
    assert days.to_period("M").is_unique


def test_ng_rule_counts_business_days_on_the_trading_calendar():
    # Feb-2026 contract: with a full calendar, last trade is Wed 28 Jan and the
    # roll is Thu 29 Jan. Remove Thu 29 Jan as a holiday and the third trading
    # day before 1 Feb becomes Tue 27 Jan, so the roll moves to Wed 28 Jan.
    idx = pd.bdate_range("2025-12-01", "2026-02-28")
    assert pd.Timestamp("2026-01-29") in calendar_roll_days(idx, "nymex_ng")
    holiday = idx.drop(pd.Timestamp("2026-01-29"))
    days = calendar_roll_days(holiday, "nymex_ng")
    assert pd.Timestamp("2026-01-28") in days
    assert pd.Timestamp("2026-01-30") not in days


def test_bz_rule_is_the_first_trading_day_of_each_month():
    idx = pd.bdate_range("2023-11-01", "2024-03-31").drop(pd.Timestamp("2024-01-01"))
    days = calendar_roll_days(idx, "BZ=F")
    assert list(days.strftime("%Y-%m-%d")) == ["2023-12-01", "2024-01-02", "2024-02-01", "2024-03-01"]


def test_first_date_of_the_sample_is_never_a_roll_day():
    idx = pd.bdate_range("2024-01-01", "2024-06-30")   # 1 Jan is a BZ month start
    assert idx[0] not in calendar_roll_days(idx, "BZ=F")


def test_period_index_is_accepted():
    idx = pd.bdate_range("2024-01-01", "2024-06-30")
    days = calendar_roll_days(pd.PeriodIndex(idx, freq="D"), "BZ=F")
    assert len(days) == 5


def test_unknown_instrument_raises_and_points_at_none():
    with pytest.raises(ValueError, match="roll_handling='none'"):
        calendar_roll_days(pd.bdate_range("2024-01-01", "2024-03-31"), "CL=F")


def test_roll_day_report_separates_rolls_from_genuine_gaps(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls, spread_scale=0.4)
    rep = roll_day_report(o, c, ng_rolls, threshold=0.15, split=TRAIN_END)
    assert rep["n_roll_days"] == len(ng_rolls)
    assert rep["n_train"] + rep["n_after_split"] == len(ng_rolls)
    assert rep["n_large_off_roll"] == 0          # all large gaps are planted rolls
    assert rep["mean_abs_gap_roll"] > 10 * rep["mean_abs_gap_other"]


# ---------------------------------------------------------------------------
# Returns and proxies on roll days
# ---------------------------------------------------------------------------

def test_roll_adjusted_return_is_open_to_close_on_roll_days_only(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    r = roll_adjusted_returns(o, c, ng_rolls)
    raw = np.log(c / c.shift(1)).dropna()
    assert r.index.equals(raw.index)
    on_roll = r.index.isin(ng_rolls)
    np.testing.assert_allclose(r[on_roll], np.log(c / o)[r.index[on_roll]])
    np.testing.assert_allclose(r[~on_roll], raw[~on_roll])
    assert not np.allclose(r[on_roll], raw[on_roll])   # the planted spread was removed


def test_overnight_term_is_zero_on_roll_days(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    on = overnight_variance(o, c, ng_rolls)
    assert np.isnan(on.iloc[0])                       # still no prior close
    assert (on[on.index.isin(ng_rolls)] == 0.0).all()
    np.testing.assert_allclose(on[~on.index.isin(ng_rolls)].iloc[1:],
                               overnight_variance(o, c)[~on.index.isin(ng_rolls)].iloc[1:])
    gkon = garman_klass_overnight(o, h, l, c, ng_rolls)
    gk = garman_klass(o, h, l, c)
    np.testing.assert_allclose(gkon[gkon.index.isin(ng_rolls)], gk[gk.index.isin(ng_rolls)])


def test_compute_proxy_passes_roll_days_to_overnight_proxies_only(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    r = np.log(c / c.shift(1)).dropna()
    kw = dict(returns=r, open_=o, high=h, low=l, close=c)
    with_r = compute_proxy("garman_klass_overnight", roll_days=ng_rolls, **kw)
    without = compute_proxy("garman_klass_overnight", **kw)
    assert (with_r <= without + 1e-15).all() and (with_r < without).any()
    # intraday proxies never read the gap
    pd.testing.assert_series_equal(compute_proxy("garman_klass", roll_days=ng_rolls, **kw),
                                   compute_proxy("garman_klass", **kw))


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

def test_floor_is_fitted_on_the_training_window_only():
    idx = pd.bdate_range("2017-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    s = pd.Series(rng.lognormal(-7, 1, len(idx)), index=idx)
    floored, floor = floor_proxy(s, 0.01, TRAIN_END)
    s2 = s.copy()
    s2[s2.index > TRAIN_END] = s2[s2.index > TRAIN_END] * 1e-3   # wreck the test period
    _, floor2 = floor_proxy(s2, 0.01, TRAIN_END)
    assert floor == floor2
    assert np.isclose(floor, np.quantile(s[s.index <= TRAIN_END], 0.01))
    assert (floored >= floor).all()
    pd.testing.assert_series_equal(floored[s >= floor], s[s >= floor])


def test_floor_ignores_non_positive_training_values():
    # >1% zeros (zero-range OHLC records): a plain 1% quantile would be 0.
    idx = pd.bdate_range("2017-01-01", "2018-12-31")
    rng = np.random.default_rng(2)
    v = rng.lognormal(-7, 1, len(idx))
    v[:20] = 0.0
    v[20:23] = -1e-6
    s = pd.Series(v, index=idx)
    assert np.quantile(s, 0.01) <= 0.0
    floored, floor = floor_proxy(s, 0.01, TRAIN_END)
    assert floor > 0 and np.isclose(floor, np.quantile(v[v > 0], 0.01))
    assert (floored > 0).all()


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1])
def test_floor_rejects_quantiles_outside_the_unit_interval(q):
    s = pd.Series([1e-4, 2e-4], index=pd.bdate_range("2018-01-01", periods=2))
    with pytest.raises(ValueError):
        floor_proxy(s, q, TRAIN_END)


def test_model_floors_at_zero_are_no_ops_on_a_floored_series():
    # With the data-layer floor in place, target_floor_q=0.0 must leave the
    # target untouched, so the pipeline has exactly one floor.
    rng = np.random.default_rng(3)
    s = pd.Series(rng.lognormal(-7, 1, 500), index=pd.bdate_range("2017-01-01", periods=500))
    floored, _ = floor_proxy(s, 0.01, TRAIN_END)
    np.testing.assert_array_equal(log_variance_target(floored.to_numpy(), 0.0),
                                  np.log(floored.to_numpy()))


def test_lstm_target_floor_defaults_to_winsor_limit_and_can_defer():
    assert LSTMVolatilityModel(device="cpu")._target_floor_q() == 0.01
    assert LSTMVolatilityModel(device="cpu", winsor_limits=(0.02, 0.02))._target_floor_q() == 0.02
    assert LSTMVolatilityModel(device="cpu", target_floor_q=0.0)._target_floor_q() == 0.0
    assert LSTMHybridModel(device="cpu", target_floor_q=0.0)._target_floor_q() == 0.0


# ---------------------------------------------------------------------------
# prepare_series: one rule for every series
# ---------------------------------------------------------------------------

PROXIES = ["garman_klass_overnight", "rogers_satchell_overnight", "garman_klass"]


@pytest.mark.parametrize("mode", ["adjust", "drop"])
def test_cleaned_series_do_not_depend_on_the_spread(calendar, ng_rolls, mode):
    small = prepare_series(*make_ohlc(calendar, ng_rolls, spread_scale=0.0),
                           proxies=PROXIES, train_end=TRAIN_END, roll_rule="NG=F",
                           roll_handling=mode, floor_q=None)
    large = prepare_series(*make_ohlc(calendar, ng_rolls, spread_scale=0.5),
                           proxies=PROXIES, train_end=TRAIN_END, roll_rule="NG=F",
                           roll_handling=mode, floor_q=None)
    pd.testing.assert_series_equal(small.returns, large.returns, rtol=1e-10)
    for name in [*PROXIES, "squared"]:
        pd.testing.assert_series_equal(small.proxies[name], large.proxies[name], rtol=1e-10)


def test_untreated_series_do_depend_on_the_spread(calendar, ng_rolls):
    # The test above has teeth only if the untreated pipeline fails it.
    kw = dict(proxies=PROXIES, train_end=TRAIN_END, roll_rule="NG=F",
              roll_handling="none", floor_q=None)
    small = prepare_series(*make_ohlc(calendar, ng_rolls, spread_scale=0.0), **kw)
    large = prepare_series(*make_ohlc(calendar, ng_rolls, spread_scale=0.5), **kw)
    assert not np.allclose(small.returns, large.returns)
    assert not np.allclose(small.proxies["garman_klass_overnight"],
                           large.proxies["garman_klass_overnight"])


def test_adjust_mode_builds_one_consistent_set(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    ps = prepare_series(o, h, l, c, proxies=["garman_klass_overnight", "garman_klass_overnight"],
                        train_end=TRAIN_END, roll_rule="NG=F", roll_handling="adjust", floor_q=None)
    assert list(ps.proxies) == ["garman_klass_overnight", "squared"]
    for p in ps.proxies.values():
        assert p.index.equals(ps.returns.index)
    on_roll = ps.returns.index.isin(ng_rolls)
    np.testing.assert_allclose(ps.returns[on_roll], np.log(c / o)[ps.returns.index[on_roll]])
    np.testing.assert_allclose(ps.proxies["squared"], ps.returns ** 2)
    np.testing.assert_allclose(ps.proxies["garman_klass_overnight"][on_roll],
                               garman_klass(o, h, l, c)[ps.returns.index[on_roll]])


def test_drop_mode_computes_returns_before_removing_the_roll_row(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls, spread_scale=0.5)
    ps = prepare_series(o, h, l, c, proxies=["garman_klass_overnight"], train_end=TRAIN_END,
                        roll_rule="NG=F", roll_handling="drop", floor_q=None)
    raw = np.log(c / c.shift(1)).dropna()
    assert len(ps.returns) == len(raw) - len(ng_rolls)
    assert not ps.returns.index.isin(ng_rolls).any()
    for p in ps.proxies.values():
        assert not p.index.isin(ng_rolls).any()
    # The day after a roll keeps ln(C_{t+1}/C_t): both prices from the new
    # contract. Dropping the PRICE instead would give ln(C_{t+1}/C_{t-1}),
    # which spans the contract switch again.
    pos = {d: i for i, d in enumerate(calendar)}
    for d in ng_rolls[:10]:
        nxt = calendar[pos[d] + 1]
        assert np.isclose(ps.returns[nxt], np.log(c[nxt] / c[d]))
        assert not np.isclose(ps.returns[nxt], np.log(c[nxt] / c[calendar[pos[d] - 1]]))


def test_none_mode_reproduces_the_untreated_pipeline(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    ps = prepare_series(o, h, l, c, proxies=["garman_klass_overnight"], train_end=TRAIN_END,
                        roll_handling="none", floor_q=None)
    r = np.log(c / c.shift(1)).dropna()
    pd.testing.assert_series_equal(ps.returns, r.rename("returns"))
    pd.testing.assert_series_equal(
        ps.proxies["garman_klass_overnight"],
        compute_proxy("garman_klass_overnight", returns=r, open_=o, high=h, low=l, close=c),
    )
    assert ps.roll_report is None and len(ps.roll_days) == 0


def test_prepare_series_floors_every_proxy_on_training_data(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    ps = prepare_series(o, h, l, c, proxies=["garman_klass_overnight"], train_end=TRAIN_END,
                        roll_rule="NG=F", roll_handling="adjust", floor_q=0.01)
    for name, p in ps.proxies.items():
        assert ps.floors[name] > 0
        assert (p >= ps.floors[name]).all()
    d = ps.diagnostics
    assert d.attrs["roll_days_train"] + d.attrs["roll_days_after"] == len(ps.roll_days)
    assert int(d.loc["garman_klass_overnight", "floored_train"]) >= 1


def test_prepare_series_validates_its_arguments(calendar, ng_rolls):
    o, h, l, c = make_ohlc(calendar, ng_rolls)
    with pytest.raises(ValueError, match="roll_rule"):
        prepare_series(o, h, l, c, proxies=[], train_end=TRAIN_END, roll_handling="adjust")
    with pytest.raises(ValueError, match="roll_handling"):
        prepare_series(o, h, l, c, proxies=[], train_end=TRAIN_END, roll_rule="NG=F",
                       roll_handling="exclude")
    with pytest.raises(ValueError, match="no contract-roll rule"):
        prepare_series(o, h, l, c, proxies=[], train_end=TRAIN_END, roll_rule="CL=F")
    # 'none' needs no calendar: an unknown instrument just has no roll report
    ps = prepare_series(o, h, l, c, proxies=[], train_end=TRAIN_END, roll_rule="CL=F",
                        roll_handling="none")
    assert ps.roll_report is None


def test_to_period_index_keeps_values(calendar, ng_rolls):
    ps = prepare_series(*make_ohlc(calendar, ng_rolls), proxies=["garman_klass_overnight"],
                        train_end=TRAIN_END, roll_rule="NG=F")
    pp = ps.to_period_index()
    assert isinstance(pp.returns.index, pd.PeriodIndex)
    np.testing.assert_array_equal(pp.returns.to_numpy(), ps.returns.to_numpy())
    assert all(isinstance(v.index, pd.PeriodIndex) for v in pp.proxies.values())
