"""
Tests for the statistical-test layer the article reports (specification of
October 2026):

  * DM only against the benchmark GARCH(1,1)-N, one column per model;
  * the MCS block length stated explicitly and shared by both implementations;
  * the PIT decided by Anderson-Darling with a Monte Carlo p-value, KS kept as a
    supplementary test;
  * the Optuna hold-outs provably chronological for both ML families;
  * tuned hyperparameters carried back on the results;
  * re-scoring and date exclusion for the robustness columns.
"""
from functools import partial

import numpy as np
import pandas as pd
import pytest

# The package must be imported BEFORE torch: on macOS the xgboost and torch
# wheels each bundle libomp, and if torch's copy loads first every XGBoost fit
# in the process crashes (see models/__init__.py).
import volatility_pipeline.models.lstm_models as lstm_mod
import volatility_pipeline.models.xgb_models as xgb_mod
from volatility_pipeline.evaluation import (
    ForecastResult,
    RollingEvaluator,
    anderson_darling_uniform_test,
    arch_mcs,
    dm_vs_benchmark,
    exclude_dates,
    hyperparameter_table,
    mcs,
    rescore,
    residual_diagnostics_table,
    residual_report,
    resolve_block_size,
)
from volatility_pipeline.evaluation.normality import _ad_uniform_null
from volatility_pipeline.models import (
    GARCHModel,
    LSTMVolatilityModel,
    XGBVolatilityModel,
    make_garch,
    tuned_factory,
)
from volatility_pipeline.models.tuning import chrono_split

import torch  # noqa: E402  (after the package, see the note above)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _garch_path(n, seed, df=None):
    rng = np.random.default_rng(seed)
    h = np.empty(n); r = np.empty(n); h[0] = 1e-4
    for t in range(n):
        if t:
            h[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        z = rng.standard_t(df) * np.sqrt((df - 2) / df) if df else rng.standard_normal()
        r[t] = np.sqrt(h[t]) * z
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(r, index=idx), pd.Series(h, index=idx)


@pytest.fixture(scope="module")
def series():
    return _garch_path(700, seed=11)


def _result(name, forecasts, actuals):
    return ForecastResult(name=name, forecasts=forecasts, actuals=actuals)


@pytest.fixture(scope="module")
def results(series):
    r, h = series
    test = slice(400, None)
    proxy = (r ** 2)[test]
    truth = h[test]
    noise = np.random.default_rng(3).lognormal(0, 0.05, len(truth))
    return {
        "GARCH-NORMAL": _result("GARCH-NORMAL", truth, proxy),
        "close": _result("close", truth * noise, proxy),
        "too-high": _result("too-high", truth * 2.5, proxy),
    }


# ---------------------------------------------------------------------------
# DM against the benchmark
# ---------------------------------------------------------------------------

def test_dm_vs_benchmark_has_one_row_per_other_model(results):
    out = dm_vs_benchmark(results, benchmark="GARCH-NORMAL", loss="qlike")
    assert list(out.index) == ["close", "too-high"]
    assert list(out.columns) == ["mean_loss", "loss_diff", "dm_stat", "p_value", "verdict"]
    assert out.attrs["benchmark"] == "GARCH-NORMAL"


def test_dm_vs_benchmark_signs_and_verdicts(results):
    out = dm_vs_benchmark(results, loss="qlike")
    bad = out.loc["too-high"]
    assert bad["loss_diff"] > 0 and bad["dm_stat"] > 0       # positive = model worse
    assert bad["verdict"] == "worse" and bad["p_value"] < 0.05
    base = results["GARCH-NORMAL"].loss_series("qlike").mean()
    assert np.isclose(bad["mean_loss"] - base, bad["loss_diff"])


def test_dm_vs_benchmark_refuses_a_missing_benchmark_or_unpaired_losses(results):
    with pytest.raises(KeyError):
        dm_vs_benchmark(results, benchmark="GARCH-T")
    shifted = dict(results)
    r = results["close"]
    shifted["close"] = _result("close", r.forecasts.iloc[1:], r.actuals.iloc[1:])
    with pytest.raises(ValueError, match="different dates"):
        dm_vs_benchmark(shifted)


# ---------------------------------------------------------------------------
# MCS block length
# ---------------------------------------------------------------------------

def test_block_size_rules():
    assert resolve_block_size(1004) == 31 == resolve_block_size(1004, "sqrt")
    assert resolve_block_size(1004, "cbrt") == 10
    assert resolve_block_size(1004, 7) == 7
    for bad in (0, -3, "auto", 2.5):
        with pytest.raises(ValueError):
            resolve_block_size(1004, bad)


def test_both_mcs_implementations_use_the_same_stated_block_length(results):
    own = mcs(results, loss="qlike", alpha=0.05, n_boot=200, seed=1)
    ext = arch_mcs(results, loss="qlike", size=0.05, n_boot=200, seed=1)
    T = len(results["GARCH-NORMAL"].forecasts)
    assert own.block_size == ext.attrs["block_size"] == int(np.sqrt(T))
    own7 = mcs(results, loss="qlike", n_boot=200, block_size=7)
    ext7 = arch_mcs(results, loss="qlike", n_boot=200, block_size=7)
    assert own7.block_size == ext7.attrs["block_size"] == 7
    assert "block_size=7" in repr(own7)


# ---------------------------------------------------------------------------
# PIT: Anderson-Darling with a Monte Carlo p-value
# ---------------------------------------------------------------------------

def test_ad_monte_carlo_null_matches_the_asymptotic_critical_value():
    null = _ad_uniform_null(1000, 5000, 7)
    assert abs(np.quantile(null, 0.95) - 2.492) < 0.15
    assert _ad_uniform_null(1000, 5000, 7) is null          # cached, built once


def test_ad_test_holds_its_size_under_the_null():
    rng = np.random.default_rng(5)
    rejections = [anderson_darling_uniform_test(rng.random(400), n_mc=2000)["ad_pval"] < 0.05
                  for _ in range(300)]
    assert 0.02 <= np.mean(rejections) <= 0.09


def test_ad_test_rejects_a_wrong_distribution_at_the_floor():
    u = np.random.default_rng(6).random(1500) ** 3
    out = anderson_darling_uniform_test(u, n_mc=2000)
    assert out["ad_pval"] == pytest.approx(1 / 2001)
    with pytest.raises(ValueError):
        anderson_darling_uniform_test(np.linspace(-0.5, 0.5, 100))


@pytest.fixture(scope="module")
def fitted_models():
    r, _ = _garch_path(1500, seed=21, df=5)          # fat-tailed innovations
    return {"GARCH-NORMAL": GARCHModel("GARCH", "normal").fit(r),
            "GARCH-T": GARCHModel("GARCH", "t").fit(r)}


def test_residual_table_puts_pit_ad_in_the_main_block(fitted_models):
    tbl = residual_diagnostics_table(fitted_models, n_boot=100)
    main, supp = tbl.attrs["main_columns"], tbl.attrs["supplementary_columns"]
    assert list(tbl.columns) == main + supp
    assert {"LB(20) p", "LB2(20) p", "ARCH-LM p", "PIT AD", "PIT AD p"} <= set(main)
    assert "PIT KS p" in supp and "PIT KS p" not in main
    # t innovations: the t fit passes the PIT, the Normal fit is rejected
    assert tbl.loc["GARCH-T", "PIT AD p"] > 0.05 > tbl.loc["GARCH-NORMAL", "PIT AD p"]


def test_residual_report_lists_ad_first_and_ks_as_supplementary(fitted_models):
    m = fitted_models["GARCH-T"]
    rep = residual_report(m.std_resid, name="GARCH-T", pit=m.pit(), dist_label="t", n_boot=100)
    pit = rep["pit"]
    assert pit.iloc[0]["test"].startswith("Anderson-Darling") and pit.iloc[0]["role"] == "main"
    assert np.isfinite(pit.iloc[0]["p_value"])
    assert set(pit.iloc[1:]["role"]) == {"supplementary"}
    assert set(rep["normality"]["role"]) == {"supplementary"}
    assert set(rep["adequacy"]["role"]) == {"main"}


# ---------------------------------------------------------------------------
# Optuna hold-outs are chronological
# ---------------------------------------------------------------------------

def test_chrono_split_formulas():
    assert chrono_split(1000, train_frac=0.8) == 800
    assert chrono_split(1003, train_frac=0.8) == int(1003 * 0.8)
    assert chrono_split(1000, val_frac=0.15) == 850
    assert chrono_split(10, val_frac=0.01) == 9               # at least one row validates
    for kw in ({}, {"train_frac": 0.8, "val_frac": 0.2}, {"train_frac": 1.0}):
        with pytest.raises(ValueError):
            chrono_split(100, **kw)
    with pytest.raises(ValueError):
        chrono_split(1, train_frac=0.8)


def test_xgb_search_trains_on_earlier_rows_and_scores_later_ones(monkeypatch):
    seen = {"fit": [], "predict": []}

    class Spy:
        def __init__(self, **kw): pass
        def fit(self, X, y): seen["fit"].append(X[:, 0].copy()); return self
        def predict(self, X): seen["predict"].append(X[:, 0].copy()); return np.zeros(len(X))

    monkeypatch.setattr(xgb_mod.xgb, "XGBRegressor", Spy)
    n = 500
    X = np.column_stack([np.arange(n), np.random.default_rng(0).normal(size=(n, 3))])
    xgb_mod._optuna_tune(X, np.zeros(n), n_trials=3, seed=0)
    assert len(seen["fit"]) == len(seen["predict"]) == 3
    for tr, va in zip(seen["fit"], seen["predict"]):
        assert tr.max() < va.min()                       # validation strictly later
        assert len(va) == n - int(0.8 * n)               # the last 20%


def test_lstm_search_trains_on_earlier_rows_and_scores_later_ones(monkeypatch):
    seen = {"fit": [], "predict": []}

    def fake_fit(X_tr, y_tr, **kw):
        seen["fit"].append(X_tr[:, -1, 0].copy())
        return None

    def fake_predict(net, X_val, device, batch_size=512):
        seen["predict"].append(X_val[:, -1, 0].copy())
        return np.zeros(len(X_val))

    monkeypatch.setattr(lstm_mod, "_fit_network", fake_fit)
    monkeypatch.setattr(lstm_mod, "_predict_batch", fake_predict)
    n, L = 300, 10
    X = np.zeros((n, L, 2), dtype=np.float32)
    X[:, :, 0] = np.arange(n)[:, None] - np.arange(L)[::-1]    # last step of row k is day k
    lstm_mod._optuna_tune_lstm(X, np.zeros(n, dtype=np.float32), n_trials=2, seed=0,
                               max_epochs=1, patience=1, val_fraction=0.15,
                               device=torch.device("cpu"))
    for tr, va in zip(seen["fit"], seen["predict"]):
        assert tr.max() < va.min()
        assert len(va) == n - int(0.8 * n)


def test_lstm_early_stopping_tail_goes_through_the_chronological_split(monkeypatch):
    calls = []
    real = lstm_mod.chrono_split
    monkeypatch.setattr(lstm_mod, "chrono_split",
                        lambda n, **kw: (calls.append(kw), real(n, **kw))[1])
    X = np.random.default_rng(0).normal(size=(120, 5, 2)).astype(np.float32)
    lstm_mod._fit_network(X, np.zeros(120, dtype=np.float32), n_features=2, hidden_size=4,
                          num_layers=1, dropout=0.0, lr=1e-3, max_epochs=1, patience=1,
                          batch_size=32, val_fraction=0.15, seed=0, device=torch.device("cpu"))
    assert calls == [{"val_frac": 0.15}]


# ---------------------------------------------------------------------------
# Tuned hyperparameters travel back on the results
# ---------------------------------------------------------------------------

def _split(series_pair):
    r, _ = series_pair
    return r.iloc[:600], r.iloc[600:]


def test_meta_records_the_tuned_xgb_settings(series):
    train, test = _split(series)
    ev = RollingEvaluator(refit_every=25)
    res = ev.evaluate(tuned_factory(XGBVolatilityModel, tune="first", n_lags=10, n_trials=2),
                      "XGB", train, test)
    m = res.meta
    assert m["n_refits"] == len(res.refit_indices) and m["tune"] == "first"
    assert m["n_searches"] == 1
    hp = m["hyperparameters"]
    assert hp["n_lags"] == 10 and hp["use_returns"] is True and "max_depth" in hp


def test_meta_survives_the_parallel_path(series):
    train, test = _split(series)
    ev = RollingEvaluator(refit_every=50)
    specs = [(tuned_factory(XGBVolatilityModel, tune="first", n_lags=10, n_trials=2), "XGB"),
             (partial(make_garch, "GARCH", "normal"), "GARCH-NORMAL")]
    out = ev.evaluate_many(specs, train, test, n_jobs=2, verbose=False)
    assert out["XGB"].meta["n_searches"] == 1
    assert "hyperparameters" not in out["GARCH-NORMAL"].meta     # nothing to report
    tbl = hyperparameter_table(out)
    assert list(tbl.index) == ["XGB"] and tbl.loc["XGB", "n_lags"] == 10


def test_meta_records_the_lstm_settings_including_the_fixed_lookback(series):
    train, test = _split(series)
    res = RollingEvaluator(refit_every=50).evaluate(
        tuned_factory(LSTMVolatilityModel, tune="first", lookback=10, n_trials=2,
                      max_epochs=3, patience=1, device="cpu"),
        "LSTM", train, test)
    hp = res.meta["hyperparameters"]
    assert hp["lookback"] == 10 and {"hidden_size", "lr", "max_epochs"} <= set(hp)


# ---------------------------------------------------------------------------
# Robustness columns: re-scoring and date exclusion
# ---------------------------------------------------------------------------

def test_rescore_changes_only_the_yardstick(results, series):
    r, h = series
    new = rescore(results, h, proxy="true_variance")
    for name in results:
        pd.testing.assert_series_equal(new[name].forecasts, results[name].forecasts)
        assert new[name].proxy == "true_variance"
        np.testing.assert_allclose(new[name].actuals, h.reindex(new[name].forecasts.index))
    with pytest.raises(ValueError, match="miss"):
        rescore(results, h.iloc[:450])


def test_exclude_dates_works_across_index_types(results):
    dates = results["close"].forecasts.index[[0, 5, 10]]              # DatetimeIndex
    out = exclude_dates(results, dates)
    assert len(out["close"].forecasts) == len(results["close"].forecasts) - 3
    assert out["close"].meta["dates_dropped"] == 3
    per = {n: _result(n, r.forecasts.set_axis(pd.PeriodIndex(r.forecasts.index, freq="D")),
                      r.actuals.set_axis(pd.PeriodIndex(r.actuals.index, freq="D")))
           for n, r in results.items()}
    out_p = exclude_dates(per, dates)                                   # PeriodIndex results
    assert len(out_p["close"].forecasts) == len(per["close"].forecasts) - 3
    kept = out_p["close"]
    assert np.isclose(kept.metrics()["QLIKE"],
                      np.mean(np.log(kept.forecasts) + kept.actuals / kept.forecasts))
