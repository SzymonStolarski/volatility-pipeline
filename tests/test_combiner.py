"""
Tests for the GARCH-family combiner (the article's hybrid, specification of
October 2026) and its equal-weight benchmark.

The combiner is the features-mode hybrid fed with the one-step forecasts of
every GARCH specification instead of one. Three properties matter:

1. Training rows use each specification's IN-SAMPLE conditional variance
   h_{i+1|i}, known at the end of day i (parameters from the training window),
   and the target is day i+1: `test_design_rows_use_tomorrows_filtered_variance`.
2. At forecast time the combiner is fed exactly the one-step forecasts the
   standalone GARCH models produce on the same window:
   `test_combiner_is_fed_the_standalone_garch_forecasts`.
3. With a single specification nothing changes for the existing hybrids (checked
   bit for bit against the previous implementation when this was written, and
   pinned here through the single-spec equivalence).
"""
from functools import partial
import pickle

import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.evaluation import (
    ForecastResult,
    RollingEvaluator,
    equal_weight_combination,
)
from volatility_pipeline.models import (
    GARCHInputs,
    LSTMHybridModel,
    XGBHybridModel,
    make_garch,
    tuned_factory,
)
from volatility_pipeline.models.garch_models import resolve_garch_specs
from volatility_pipeline.models.xgb_models import _hybrid_design

SPECS = [("GARCH", "normal"), ("GJR-GARCH", "t"), ("EGARCH", "normal")]
N_TRAIN = 800


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(8)
    n = 900
    h = np.empty(n); r = np.empty(n); h[0] = 1e-4
    for t in range(n):
        if t:
            h[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    idx = pd.bdate_range("2014-01-01", periods=n)
    ret = pd.Series(r, index=idx)
    proxy = (ret ** 2 * np.exp(rng.normal(0, 0.3, n))).rename("proxy")
    return ret, proxy


# ---------------------------------------------------------------------------
# The GARCH inputs
# ---------------------------------------------------------------------------

def test_spec_resolution():
    assert resolve_garch_specs(None, "GJR-GARCH", "t") == [("GJR-GARCH", "t")]
    assert resolve_garch_specs([["GARCH", "normal"]]) == [("GARCH", "normal")]
    for bad in ([], [("GARCH", "skewt")], [("HARCH", "normal")], [("GARCH", "normal")] * 2):
        with pytest.raises(ValueError):
            resolve_garch_specs(bad)


def test_garch_inputs_match_standalone_models(data):
    ret, _ = data
    tr = ret.iloc[:N_TRAIN]
    inputs = GARCHInputs(SPECS).fit(tr)
    alone = [make_garch(t, d).fit(tr) for t, d in SPECS]
    np.testing.assert_array_equal(
        inputs.insample_matrix(),
        np.column_stack([m.insample_variance().to_numpy() for m in alone]),
    )
    np.testing.assert_array_equal(inputs.forecasts(), [m.forecast_variance(1)[0] for m in alone])
    inputs.update(ret.iloc[:N_TRAIN + 9])
    for m in alone:
        m.update(ret.iloc[:N_TRAIN + 9])
    np.testing.assert_array_equal(inputs.forecasts(), [m.forecast_variance(1)[0] for m in alone])
    assert inputs.names == ["GARCH-NORMAL", "GJR-GARCH-T", "EGARCH-NORMAL"]


def test_asym_order_reaches_egarch(data):
    ret, _ = data
    tr = ret.iloc[:N_TRAIN]
    assert GARCHInputs([("EGARCH", "normal")]).fit(tr).models[0].asym_order == 1
    assert GARCHInputs([("EGARCH", "normal")], asym_order=0).fit(tr).models[0].asym_order == 0
    # GARCH / GJR keep the order that defines them
    assert GARCHInputs([("GJR-GARCH", "t")], asym_order=0).fit(tr).models[0].asym_order == 1


# ---------------------------------------------------------------------------
# Training rows
# ---------------------------------------------------------------------------

def test_design_rows_use_tomorrows_filtered_variance():
    n, L, K = 40, 5, 3
    r = np.arange(n, dtype=float) / 100
    sq = r ** 2
    g = np.arange(n * K, dtype=float).reshape(n, K) + 1.0        # g[t, k] = h^(k)_{t|t-1}
    y = np.arange(n, dtype=float) + 1000.0
    X, yy = _hybrid_design(sq, r, g, y, L, True, "features")
    assert X.shape == (n - 1 - L, 2 * L + K)
    for k in range(len(X)):
        i = L + k
        np.testing.assert_array_equal(X[k, :L], sq[i - L + 1 : i + 1][::-1])
        np.testing.assert_array_equal(X[k, L:2 * L], r[i - L + 1 : i + 1][::-1])
        np.testing.assert_array_equal(X[k, 2 * L:], g[i + 1])           # known at the end of day i
        assert yy[k] == y[i + 1]                                        # target is day i+1
    _, y_res = _hybrid_design(sq, r, g[:, :1], y, L, True, "residual")
    np.testing.assert_array_equal(y_res, y[L + 1:] - g[L + 1:, 0])


# ---------------------------------------------------------------------------
# The combiners
# ---------------------------------------------------------------------------

def test_xgb_combiner_has_one_feature_per_specification(data):
    ret, proxy = data
    m = XGBHybridModel(garch_specs=SPECS, n_lags=10).fit(ret.iloc[:N_TRAIN], proxy.iloc[:N_TRAIN])
    assert m.is_combiner and "combiner of 3" in repr(m)
    assert m._xgb.n_features_in_ == 2 * 10 + 3
    assert m.feature_names()[-3:] == ["fc_GARCH-NORMAL", "fc_GJR-GARCH-T", "fc_EGARCH-NORMAL"]
    assert m.hyperparameters()["garch_inputs"] == 3
    fc = m.forecast_variance(1)[0]
    assert np.isfinite(fc) and fc > 0


def test_lstm_combiner_has_one_channel_per_specification(data):
    ret, proxy = data
    m = LSTMHybridModel(garch_specs=SPECS, lookback=10, max_epochs=3, patience=1,
                        device="cpu").fit(ret.iloc[:N_TRAIN], proxy.iloc[:N_TRAIN])
    assert m.is_combiner
    assert m._last_window.shape == (10, 2 + 3)
    # the GARCH channels are today's forecasts, constant across the window
    assert np.allclose(m._last_window[:, 2:], m._last_window[-1, 2:])
    assert m.feature_names() == ["return", "sq_return", "fc_GARCH-NORMAL",
                                 "fc_GJR-GARCH-T", "fc_EGARCH-NORMAL"]
    assert m.hyperparameters()["garch_inputs"] == 3


@pytest.mark.parametrize("cls, kw", [
    (XGBHybridModel, dict(n_lags=10)),
    (LSTMHybridModel, dict(lookback=10, max_epochs=3, patience=1, device="cpu")),
])
def test_combiner_is_fed_the_standalone_garch_forecasts(data, cls, kw):
    ret, proxy = data
    m = cls(garch_specs=SPECS, **kw).fit(ret.iloc[:N_TRAIN], proxy.iloc[:N_TRAIN])
    alone = [make_garch(t, d).fit(ret.iloc[:N_TRAIN]) for t, d in SPECS]
    np.testing.assert_array_equal(m.garch_forecasts().to_numpy(),
                                  [g.forecast_variance(1)[0] for g in alone])
    m.update(ret.iloc[:N_TRAIN + 12])
    for g in alone:
        g.update(ret.iloc[:N_TRAIN + 12])
    np.testing.assert_array_equal(m.garch_forecasts().to_numpy(),
                                  [g.forecast_variance(1)[0] for g in alone])


@pytest.mark.parametrize("cls", [XGBHybridModel, LSTMHybridModel])
def test_residual_mode_needs_exactly_one_specification(cls):
    with pytest.raises(ValueError, match="exactly one"):
        cls(garch_specs=SPECS, mode="residual")
    cls(garch_specs=SPECS[:1], mode="residual")       # one is fine


def test_single_spec_list_equals_the_single_base_signature(data):
    ret, proxy = data
    tr, tg = ret.iloc[:N_TRAIN], proxy.iloc[:N_TRAIN]
    a = XGBHybridModel(garch_model_type="GJR-GARCH", garch_dist="t", n_lags=10).fit(tr, tg)
    b = XGBHybridModel(garch_specs=[("GJR-GARCH", "t")], n_lags=10).fit(tr, tg)
    assert not b.is_combiner and b.feature_names()[-1] == "garch_fc"
    assert a.forecast_variance(1)[0] == b.forecast_variance(1)[0]


def test_combiner_tunes_once_through_the_evaluator_and_reports(data):
    ret, proxy = data
    factory = tuned_factory(XGBHybridModel, tune="first", garch_specs=SPECS, n_lags=10, n_trials=2)
    pickle.dumps(factory)                                   # must survive the process pool
    res = RollingEvaluator(refit_every=50).evaluate(
        factory, "XGB-COMB", ret.iloc[:N_TRAIN], ret.iloc[N_TRAIN:], actuals_series=proxy)
    assert res.meta["n_searches"] == 1 and res.meta["hyperparameters"]["garch_inputs"] == 3
    assert np.isfinite(res.forecasts).all() and (res.forecasts > 0).all()


def test_combiner_runs_in_the_process_pool(data):
    ret, proxy = data
    specs = [(tuned_factory(XGBHybridModel, tune="never", garch_specs=SPECS, n_lags=10), "XGB-COMB"),
             (partial(make_garch, "GARCH", "normal"), "GARCH-NORMAL")]
    out = RollingEvaluator(refit_every=50).evaluate_many(
        specs, ret.iloc[:N_TRAIN], ret.iloc[N_TRAIN:], actuals_series=proxy, n_jobs=2, verbose=False)
    assert set(out) == {"XGB-COMB", "GARCH-NORMAL"}


# ---------------------------------------------------------------------------
# The equal-weight benchmark
# ---------------------------------------------------------------------------

def _res(name, fc, act):
    return ForecastResult(name=name, forecasts=fc, actuals=act, refit_indices=[0, 10])


@pytest.fixture
def members():
    idx = pd.bdate_range("2022-07-01", periods=30)
    act = pd.Series(np.linspace(1e-4, 4e-4, 30), index=idx)
    return {f"M{k}": _res(f"M{k}", pd.Series(np.full(30, (k + 1) * 1e-4), index=idx), act)
            for k in range(3)}


def test_equal_weight_is_the_mean_of_the_member_forecasts(members):
    ew = equal_weight_combination(members, ["M0", "M1", "M2"], name="EW")
    np.testing.assert_allclose(ew.forecasts, 2e-4)
    pd.testing.assert_series_equal(ew.actuals, members["M0"].actuals)
    assert ew.name == "EW" and ew.meta["members"] == ["M0", "M1", "M2"]


def test_equal_weight_validates_its_members(members):
    with pytest.raises(KeyError):
        equal_weight_combination(members, ["M0", "M9"])
    with pytest.raises(ValueError, match="at least two"):
        equal_weight_combination(members, ["M0"])
    shifted = dict(members)
    m = members["M1"]
    shifted["M1"] = _res("M1", m.forecasts.iloc[1:], m.actuals.iloc[1:])
    with pytest.raises(ValueError, match="different dates"):
        equal_weight_combination(shifted, ["M0", "M1"])
    other = dict(members)
    other["M2"] = _res("M2", m.forecasts, m.actuals * 2)
    with pytest.raises(ValueError, match="different actuals"):
        equal_weight_combination(other, ["M0", "M2"])
