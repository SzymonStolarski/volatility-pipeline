from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from .garch_models import GARCHModel
from .targets import log_variance_target, resolve_target, smearing_factor


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
    # n_jobs=1: single-threaded by default so that model-level process
    # parallelism (RollingEvaluator n_jobs > 1) does not cause oversubscription.
    # Set to -1 when evaluating a single model sequentially.
    "n_jobs":           1,
}


def _optuna_tune(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    seed: int,
    n_jobs: int = 1,
) -> dict:
    """
    Tune XGBRegressor via Optuna on an 80/20 time-series holdout split.

    Parameters
    ----------
    n_jobs : parallel Optuna workers (1 → sequential trials).
             Keep at 1 when the caller is already inside a model-level process pool.
    """
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
            "n_jobs":           1,   # always 1 inside trials; outer n_jobs handles concurrency
            "random_state":     seed,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr)
        return float(np.mean((m.predict(X_val) - y_val) ** 2))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    best = study.best_params
    best["objective"] = "reg:squarederror"
    return best


class XGBVolatilityModel:
    """
    Standalone XGBoost volatility forecaster.

    Features: n_lags of lagged squared returns; optionally also lagged raw returns.
    Target:   the next-step realized-variance proxy supplied by the caller
              (see `fit`), falling back to squared returns.
    Compatible with RollingEvaluator (.fit / .update / .forecast_variance).
    """

    def __init__(
        self,
        n_lags: int = 5,
        use_returns: bool = True,
        use_optuna: bool = False,
        n_trials: int = 50,
        optuna_n_jobs: int = 1,
        xgb_params: dict | None = None,
        seed: int = 42,
        log_target: bool = True,
        retransform: str = "smearing",
        target_floor_q: float = 0.01,
    ) -> None:
        if retransform not in ("smearing", "none"):
            raise ValueError(
                f"retransform must be 'smearing' or 'none', got {retransform!r}"
            )
        self.n_lags         = n_lags
        self.use_returns    = use_returns
        self.use_optuna     = use_optuna
        self.n_trials       = n_trials
        self.optuna_n_jobs  = optuna_n_jobs
        self.xgb_params     = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed           = seed
        self.log_target     = log_target
        self.retransform    = retransform
        self.target_floor_q = target_floor_q
        self._model: xgb.XGBRegressor | None = None
        self._last_sq: np.ndarray | None = None
        self._last_r:  np.ndarray | None = None
        self._smearing: float = 1.0

    def fit(self, returns: pd.Series, target: pd.Series | None = None) -> "XGBVolatilityModel":
        """
        `target` is the realized-variance proxy aligned to `returns` — the same
        series the evaluator scores against. None falls back to squared returns.
        """
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        y_raw = resolve_target(r, None if target is None else np.asarray(target, dtype=float))
        X, y = self._build_features(sq, r, y_raw)
        if self.log_target:
            y = log_variance_target(y, self.target_floor_q)
        params = (
            _optuna_tune(X, y, self.n_trials, self.seed, self.optuna_n_jobs)
            if self.use_optuna
            else {**self.xgb_params, "random_state": self.seed, "verbosity": 0}
        )
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X, y)
        self._smearing = (
            smearing_factor(y, self._model.predict(X))
            if (self.log_target and self.retransform == "smearing" and len(X))
            else 1.0
        )
        self.update(returns)
        return self

    def update(self, returns: pd.Series) -> "XGBVolatilityModel":
        """Refresh the lagged-feature state without refitting the booster."""
        r  = np.asarray(returns, dtype=float)
        if len(r) < self.n_lags:
            raise ValueError(
                f"update needs at least n_lags={self.n_lags} observations, got {len(r)}."
            )
        self._last_sq = (r ** 2)[-self.n_lags:].copy()
        self._last_r  = r[-self.n_lags:].copy()
        return self

    def _retransform(self, raw: float) -> float:
        if self.log_target:
            return float(np.exp(raw)) * self._smearing
        return max(float(raw), 1e-10)

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        row = list(self._last_sq[::-1])
        if self.use_returns:
            row += list(self._last_r[::-1])
        pred = self._retransform(float(self._model.predict(np.array([row]))[0]))
        return np.full(horizon, pred)

    def feature_names(self) -> list[str]:
        names = [f"sq_lag{i + 1}" for i in range(self.n_lags)]
        if self.use_returns:
            names += [f"r_lag{i + 1}" for i in range(self.n_lags)]
        return names

    def _build_features(
        self, sq: np.ndarray, r: np.ndarray, y_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(sq)
        rows, targets = [], []
        for i in range(self.n_lags, n - 1):
            row = list(sq[i - self.n_lags + 1 : i + 1][::-1])  # sq lag1..lagK
            if self.use_returns:
                row += list(r[i - self.n_lags + 1 : i + 1][::-1])
            rows.append(row)
            targets.append(y_raw[i + 1])
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
        optuna_n_jobs: int = 1,
        xgb_params: dict | None = None,
        seed: int = 42,
        log_target: bool = True,
        retransform: str = "smearing",
        target_floor_q: float = 0.01,
    ) -> None:
        if mode not in ("features", "residual"):
            raise ValueError(f"mode must be 'features' or 'residual', got {mode!r}")
        if retransform not in ("smearing", "none"):
            raise ValueError(
                f"retransform must be 'smearing' or 'none', got {retransform!r}"
            )
        self.garch_model_type = garch_model_type
        self.garch_dist       = garch_dist
        self.garch_p          = garch_p
        self.garch_q          = garch_q
        self.mode             = mode
        self.n_lags           = n_lags
        self.use_returns      = use_returns
        self.use_optuna       = use_optuna
        self.n_trials         = n_trials
        self.optuna_n_jobs    = optuna_n_jobs
        self.xgb_params       = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed             = seed
        # The log target applies to 'features' mode only: 'residual' mode is fit
        # on proxy - h, which is negative about half the time and has no log.
        self.log_target       = bool(log_target) and mode == "features"
        self.retransform      = retransform
        self.target_floor_q   = target_floor_q
        self._garch: GARCHModel | None       = None
        self._xgb: xgb.XGBRegressor | None  = None
        self._last_sq: np.ndarray | None     = None
        self._last_r:  np.ndarray | None     = None
        self._smearing: float                = 1.0

    def fit(self, returns: pd.Series, target: pd.Series | None = None) -> "XGBHybridModel":
        """
        `target` is the realized-variance proxy aligned to `returns` — the same
        series the evaluator scores against. None falls back to squared returns.
        """
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        n  = len(r)
        y_raw = resolve_target(r, None if target is None else np.asarray(target, dtype=float))

        self._garch = GARCHModel(
            self.garch_model_type, self.garch_dist, self.garch_p, self.garch_q
        )
        self._garch.fit(returns)
        # g_var[t] = h_{t|t-1}: GARCH in-sample conditional variance at each t
        g_var = self._garch.insample_variance().values

        rows, targets = [], []
        for i in range(self.n_lags, n - 1):
            # predicting y_raw[i+1]; use g_var[i+1] = h_{i+1|i} as GARCH feature
            row = list(sq[i - self.n_lags + 1 : i + 1][::-1])
            if self.use_returns:
                row += list(r[i - self.n_lags + 1 : i + 1][::-1])
            row.append(g_var[i + 1])
            rows.append(row)
            targets.append(
                y_raw[i + 1] if self.mode == "features"
                else y_raw[i + 1] - g_var[i + 1]
            )

        X = np.array(rows, dtype=float)
        y = np.array(targets, dtype=float)
        if self.log_target:
            y = log_variance_target(y, self.target_floor_q)

        params = (
            _optuna_tune(X, y, self.n_trials, self.seed, self.optuna_n_jobs)
            if self.use_optuna
            else {**self.xgb_params, "random_state": self.seed, "verbosity": 0}
        )
        self._xgb = xgb.XGBRegressor(**params)
        self._xgb.fit(X, y)
        self._smearing = (
            smearing_factor(y, self._xgb.predict(X))
            if (self.log_target and self.retransform == "smearing" and len(X))
            else 1.0
        )
        self.update(returns)
        return self

    def update(self, returns: pd.Series) -> "XGBHybridModel":
        """
        Refresh the lagged features AND the GARCH state without refitting
        either. The GARCH one-step forecast is an input feature, so leaving it
        stale would defeat the point of updating the lags.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < self.n_lags:
            raise ValueError(
                f"update needs at least n_lags={self.n_lags} observations, got {len(r)}."
            )
        if self._garch is not None:
            self._garch.update(returns)
        self._last_sq = (r ** 2)[-self.n_lags:].copy()
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
        if self.mode == "features":
            result = (
                float(np.exp(xgb_pred)) * self._smearing if self.log_target
                else max(xgb_pred, 1e-10)
            )
        else:
            result = max(garch_fc + xgb_pred, 1e-10)
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
