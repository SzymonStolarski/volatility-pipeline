"""
Both ML families must see the same information: n lagged returns and n lagged
squared returns, with n fixed by the specification (10) and never searched.

The frozen specification makes this the condition for a fair comparison of
model classes. Before this was pinned, the LSTM's lookback was an Optuna
dimension over {5, 10, 20, 40} while XGBoost used a fixed n_lags, so the LSTM
could pick a longer history on validation data than XGBoost ever saw.

`test_xgb_and_lstm_see_exactly_the_same_days` is the strongest check: for every
training row, the XGBoost feature vector and the LSTM input window must contain
the same returns and squared returns, and predict the same target.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.models import (
    LSTMHybridModel,
    LSTMVolatilityModel,
    TuningCache,
    XGBHybridModel,
    XGBVolatilityModel,
)
from volatility_pipeline.models.lstm_models import (
    LSTM_SEARCH_SPACE,
    _build_sequences,
    _optuna_tune_lstm,
)

N_LAGS = 10


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(7)
    n = 400
    h = np.empty(n); r = np.empty(n); h[0] = 1e-4
    for t in range(n):
        if t:
            h[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


def test_xgb_and_lstm_see_exactly_the_same_days(returns):
    r = returns.to_numpy()
    sq = r ** 2
    target = sq.copy()
    X_xgb, y_xgb = XGBVolatilityModel(n_lags=N_LAGS, use_returns=True)._build_features(sq, r, target)
    X_lstm, y_lstm = _build_sequences(np.column_stack([r, sq]), target, N_LAGS)
    assert len(X_xgb) == len(X_lstm)
    np.testing.assert_allclose(y_xgb, y_lstm, rtol=1e-6)
    for k in range(len(X_xgb)):
        # XGBoost stores lag1..lagK (most recent first); the LSTM window runs oldest -> newest
        np.testing.assert_allclose(X_xgb[k, :N_LAGS][::-1], X_lstm[k][:, 1], rtol=1e-6)   # r^2
        np.testing.assert_allclose(X_xgb[k, N_LAGS:][::-1], X_lstm[k][:, 0], rtol=1e-6)   # r


def test_xgb_tuning_never_changes_the_feature_set(returns):
    cache = TuningCache()
    m = XGBVolatilityModel(n_lags=N_LAGS, use_returns=True, tune="first",
                           tuning_cache=cache, n_trials=3).fit(returns)
    assert cache.n_searches == 1
    assert not {"n_lags", "use_returns"} & set(cache.params)
    assert m.n_lags == N_LAGS
    assert m._model.n_features_in_ == 2 * N_LAGS
    assert m.feature_names() == [f"sq_lag{i}" for i in range(1, N_LAGS + 1)] + \
                                [f"r_lag{i}" for i in range(1, N_LAGS + 1)]


def test_xgb_hybrid_tuning_never_changes_the_feature_set(returns):
    cache = TuningCache()
    m = XGBHybridModel(n_lags=N_LAGS, use_returns=True, tune="first",
                       tuning_cache=cache, n_trials=3).fit(returns)
    assert not {"n_lags", "use_returns"} & set(cache.params)
    assert m._xgb.n_features_in_ == 2 * N_LAGS + 1        # + the GARCH forecast


LSTM_KW = dict(n_trials=2, seed=1, max_epochs=5, patience=2, device="cpu")


@pytest.mark.parametrize("cls, kw, channels", [
    (LSTMVolatilityModel, {}, 2),                       # r, r^2
    (LSTMHybridModel, {"mode": "features"}, 3),         # r, r^2, GARCH forecast
    (LSTMHybridModel, {"mode": "residual"}, 2),
])
def test_lstm_tuning_keeps_the_lookback_fixed(returns, cls, kw, channels):
    cache = TuningCache()
    m = cls(lookback=N_LAGS, tune="first", tuning_cache=cache, **kw, **LSTM_KW).fit(returns)
    assert cache.n_searches == 1
    assert "lookback" not in cache.params
    assert m.lookback == N_LAGS
    assert m._last_window.shape == (N_LAGS, channels)
    m.update(returns)
    assert m._last_window.shape == (N_LAGS, channels)


def test_lstm_search_refuses_the_lookback():
    X = np.zeros((200, N_LAGS, 2), dtype=np.float32)
    y = np.zeros(200, dtype=np.float32)
    with pytest.raises(ValueError, match="lookback"):
        _optuna_tune_lstm(X, y, n_trials=1, seed=0, max_epochs=1, patience=1,
                          val_fraction=0.15, device="cpu",
                          search_space={**LSTM_SEARCH_SPACE, "lookback": [5, 10, 20]})
