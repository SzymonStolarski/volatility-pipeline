from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from .garch_models import GARCHModel


_DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",
    "verbosity":        0,
}


def _optuna_tune(X: np.ndarray, y: np.ndarray, n_trials: int, seed: int) -> dict:
    """Tune XGBRegressor via Optuna on an 80/20 time-series holdout split."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 2, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":        "reg:squarederror",
            "verbosity":        0,
            "random_state":     seed,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr)
        return float(np.mean((m.predict(X_val) - y_val) ** 2))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["objective"] = "reg:squarederror"
    return best


class XGBVolatilityModel:
    """
    Standalone XGBoost volatility forecaster.

    Features: n_lags of lagged squared returns; optionally also lagged raw returns.
    Target:   next-step squared return (realized variance proxy).
    Compatible with RollingEvaluator (.fit / .forecast_variance interface).
    """

    def __init__(
        self,
        n_lags: int = 5,
        use_returns: bool = True,
        use_optuna: bool = False,
        n_trials: int = 50,
        xgb_params: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.n_lags       = n_lags
        self.use_returns  = use_returns
        self.use_optuna   = use_optuna
        self.n_trials     = n_trials
        self.xgb_params   = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed         = seed
        self._model: xgb.XGBRegressor | None = None
        self._last_sq: np.ndarray | None = None
        self._last_r:  np.ndarray | None = None

    def fit(self, returns: pd.Series) -> "XGBVolatilityModel":
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        X, y = self._build_features(sq, r)
        params = (
            _optuna_tune(X, y, self.n_trials, self.seed)
            if self.use_optuna
            else {**self.xgb_params, "random_state": self.seed, "verbosity": 0}
        )
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X, y)
        self._last_sq = sq[-self.n_lags:].copy()
        self._last_r  = r[-self.n_lags:].copy()
        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        row = list(self._last_sq[::-1])
        if self.use_returns:
            row += list(self._last_r[::-1])
        pred = max(float(self._model.predict(np.array([row]))[0]), 1e-10)
        return np.full(horizon, pred)

    def feature_names(self) -> list[str]:
        names = [f"sq_lag{i + 1}" for i in range(self.n_lags)]
        if self.use_returns:
            names += [f"r_lag{i + 1}" for i in range(self.n_lags)]
        return names

    def _build_features(
        self, sq: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(sq)
        rows, targets = [], []
        for i in range(self.n_lags, n - 1):
            row = list(sq[i - self.n_lags + 1 : i + 1][::-1])  # sq lag1..lagK
            if self.use_returns:
                row += list(r[i - self.n_lags + 1 : i + 1][::-1])
            rows.append(row)
            targets.append(sq[i + 1])
        return np.array(rows, dtype=float), np.array(targets, dtype=float)

    def __repr__(self) -> str:
        return (
            f"XGBVolatilityModel(n_lags={self.n_lags}, "
            f"use_returns={self.use_returns}, use_optuna={self.use_optuna})"
        )


class XGBHybridModel:
    """
    Hybrid XGBoost + GARCH volatility model. Compatible with RollingEvaluator.

    mode='features':
        XGB predicts variance directly, using the GARCH one-step-ahead forecast
        (h_{t+1|t}) and lagged returns as features.  Final forecast = XGB output.

    mode='residual':
        XGB predicts the GARCH residual: realized_var − GARCH_forecast.
        Final forecast = GARCH_forecast + XGB_residual_correction.

    In both modes the internal GARCH model is re-estimated on every .fit() call,
    so the hybrid model is self-contained and works transparently with
    RollingEvaluator's expanding/sliding window refitting logic.
    """

    def __init__(
        self,
        garch_model_type: str = "GARCH",
        garch_dist: str = "normal",
        garch_p: int = 1,
        garch_q: int = 1,
        mode: str = "features",
        n_lags: int = 5,
        use_returns: bool = True,
        use_optuna: bool = False,
        n_trials: int = 50,
        xgb_params: dict | None = None,
        seed: int = 42,
    ) -> None:
        if mode not in ("features", "residual"):
            raise ValueError(f"mode must be 'features' or 'residual', got {mode!r}")
        self.garch_model_type = garch_model_type
        self.garch_dist       = garch_dist
        self.garch_p          = garch_p
        self.garch_q          = garch_q
        self.mode             = mode
        self.n_lags           = n_lags
        self.use_returns      = use_returns
        self.use_optuna       = use_optuna
        self.n_trials         = n_trials
        self.xgb_params       = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed             = seed
        self._garch: GARCHModel | None       = None
        self._xgb: xgb.XGBRegressor | None  = None
        self._last_sq: np.ndarray | None     = None
        self._last_r:  np.ndarray | None     = None

    def fit(self, returns: pd.Series) -> "XGBHybridModel":
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        n  = len(r)

        self._garch = GARCHModel(
            self.garch_model_type, self.garch_dist, self.garch_p, self.garch_q
        )
        self._garch.fit(returns)
        # g_var[t] = h_{t|t-1}: GARCH in-sample conditional variance at each t
        g_var = self._garch.insample_variance().values

        rows, targets = [], []
        for i in range(self.n_lags, n - 1):
            # predicting sq[i+1]; use g_var[i+1] = h_{i+1|i} as GARCH feature
            row = list(sq[i - self.n_lags + 1 : i + 1][::-1])
            if self.use_returns:
                row += list(r[i - self.n_lags + 1 : i + 1][::-1])
            row.append(g_var[i + 1])
            rows.append(row)
            targets.append(
                sq[i + 1] if self.mode == "features"
                else sq[i + 1] - g_var[i + 1]
            )

        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=float)

        params = (
            _optuna_tune(X, y, self.n_trials, self.seed)
            if self.use_optuna
            else {**self.xgb_params, "random_state": self.seed, "verbosity": 0}
        )
        self._xgb = xgb.XGBRegressor(**params)
        self._xgb.fit(X, y)
        self._last_sq = sq[-self.n_lags:].copy()
        self._last_r  = r[-self.n_lags:].copy()
        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._xgb is None or self._garch is None:
            raise RuntimeError("Call .fit() first.")
        garch_fc = float(self._garch.forecast_variance(horizon=1)[0])
        row = list(self._last_sq[::-1])
        if self.use_returns:
            row += list(self._last_r[::-1])
        row.append(garch_fc)
        xgb_pred = float(self._xgb.predict(np.array([row]))[0])
        result = (
            max(xgb_pred, 1e-10)
            if self.mode == "features"
            else max(garch_fc + xgb_pred, 1e-10)
        )
        return np.full(horizon, result)

    def feature_names(self) -> list[str]:
        names = [f"sq_lag{i + 1}" for i in range(self.n_lags)]
        if self.use_returns:
            names += [f"r_lag{i + 1}" for i in range(self.n_lags)]
        names.append("garch_fc")
        return names

    def __repr__(self) -> str:
        return (
            f"XGBHybridModel(garch={self.garch_model_type}-{self.garch_dist}, "
            f"mode={self.mode!r}, n_lags={self.n_lags}, use_optuna={self.use_optuna})"
        )
