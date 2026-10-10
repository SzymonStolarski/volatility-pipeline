from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from .garch_models import GARCHInputs, GARCHModel
from .targets import log_variance_target, resolve_target, smearing_factor
from .tuning import TuningCache, chrono_split, resolve_hyperparameters, validate_tune


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

    Only the booster's own settings are searched. The feature set — n_lags
    lagged squared returns and n_lags lagged returns — is already built into X
    and is deliberately NOT a search dimension: it is the model's information
    set, fixed by the specification and shared with the LSTM's lookback, so
    that neither ML family can choose a longer history on validation data.

    Parameters
    ----------
    n_jobs : parallel Optuna workers (1 → sequential trials).
             Keep at 1 when the caller is already inside a model-level process pool.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = chrono_split(len(X), train_frac=0.8)   # earlier 80% fit, later 20% scored
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


_HP_NOISE = ("verbosity", "n_jobs", "objective", "random_state")


def _xgb_hyperparameters(model) -> dict:
    """Booster settings used by the last fit, plus the fixed information set."""
    if model._params is None:
        raise RuntimeError("Call .fit() first.")
    hp = {k: v for k, v in model._params.items() if k not in _HP_NOISE}
    hp.update(n_lags=model.n_lags, use_returns=model.use_returns)
    return hp


class XGBVolatilityModel:
    """
    Standalone XGBoost volatility forecaster.

    Features: n_lags of lagged squared returns; with use_returns=True (the
    specification) also n_lags of lagged returns. n_lags is fixed, never tuned.
    Target:   the next-step realized-variance proxy supplied by the caller
              (see `fit`), falling back to squared returns.
    Compatible with RollingEvaluator (.fit / .update / .forecast_variance).
    """

    def __init__(
        self,
        n_lags: int = 5,
        use_returns: bool = True,
        tune: str = "never",
        tuning_cache: TuningCache | None = None,
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
        self.tune           = validate_tune(tune)
        # Deliberately NOT auto-created: under RollingEvaluator the cache must be
        # SHARED across the instances it builds per re-estimation, and a
        # per-instance one would make tune="first" behave like "always" without
        # saying so. Build factories with tuning.tuned_factory.
        self.tuning_cache   = tuning_cache
        self.n_trials       = n_trials
        self.optuna_n_jobs  = optuna_n_jobs
        self.xgb_params     = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed           = seed
        self.log_target     = log_target
        self.retransform    = retransform
        self.target_floor_q = target_floor_q
        self._model: xgb.XGBRegressor | None = None
        self._params: dict | None = None
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
        params = resolve_hyperparameters(
            self.tune, self.tuning_cache,
            lambda: _optuna_tune(X, y, self.n_trials, self.seed, self.optuna_n_jobs),
            {**self.xgb_params, "random_state": self.seed, "verbosity": 0},
        )
        self._params = dict(params)
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

    def hyperparameters(self) -> dict:
        """Settings the last fit ran with (tuned or default), for reporting."""
        return _xgb_hyperparameters(self)

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
            f"use_returns={self.use_returns}, tune={self.tune!r})"
        )


def _hybrid_design(
    sq: np.ndarray,
    r: np.ndarray,
    g_vars: np.ndarray,
    y_raw: np.ndarray,
    n_lags: int,
    use_returns: bool,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Training rows of the XGBoost hybrid / combiner.

    Row for predicting day i+1 (i = n_lags .. n-2):
      [r^2_i .. r^2_{i-n_lags+1}]  [r_i .. r_{i-n_lags+1}]  [h^(1)_{i+1|i} .. h^(K)_{i+1|i}]
    g_vars[t, k] = h^(k)_{t|t-1} is specification k's in-sample conditional
    variance, so g_vars[i+1] is known at the end of day i: the GARCH state for
    day i+1 is filtered from returns up to i. Target: y_raw[i+1] ('features'),
    or y_raw[i+1] - h^(1)_{i+1|i} ('residual', single specification only).
    """
    rows, targets = [], []
    for i in range(n_lags, len(sq) - 1):
        row = list(sq[i - n_lags + 1 : i + 1][::-1])
        if use_returns:
            row += list(r[i - n_lags + 1 : i + 1][::-1])
        row += list(g_vars[i + 1])
        rows.append(row)
        targets.append(y_raw[i + 1] if mode == "features" else y_raw[i + 1] - g_vars[i + 1, 0])
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


class XGBHybridModel:
    """
    Hybrid XGBoost + GARCH volatility model. Compatible with RollingEvaluator.

    mode='features':
        XGB predicts variance directly from lagged returns and squared returns
        plus the one-step-ahead forecast(s) of the GARCH specification(s) in
        `garch_specs`. Final forecast = XGB output.
          * one specification  -> the single-base hybrid
          * the GARCH family   -> the COMBINER: XGBoost as a nonlinear combiner
                                  of the family's one-step forecasts (the
                                  article's hybrid, specification of Oct 2026).

    mode='residual' (one specification only):
        XGB predicts the GARCH residual: realized_var - GARCH_forecast.
        Final forecast = GARCH_forecast + XGB_residual_correction.

    Training uses each specification's IN-SAMPLE conditional variance
    h_{t+1|t} (parameters estimated on the training window); forecasting uses
    its genuine one-step forecast after update(). See GARCHInputs.

    The GARCH models are re-estimated on every .fit() call, so the model is
    self-contained and works transparently with RollingEvaluator's expanding /
    sliding refitting.

    `garch_specs=None` keeps the single-base signature (garch_model_type,
    garch_dist); `asym_order` reaches EGARCH / APARCH as in make_garch.
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
        tune: str = "never",
        tuning_cache: TuningCache | None = None,
        n_trials: int = 50,
        optuna_n_jobs: int = 1,
        xgb_params: dict | None = None,
        seed: int = 42,
        log_target: bool = True,
        retransform: str = "smearing",
        target_floor_q: float = 0.01,
        garch_specs: list | None = None,
        asym_order: int | None = None,
    ) -> None:
        if mode not in ("features", "residual"):
            raise ValueError(f"mode must be 'features' or 'residual', got {mode!r}")
        if retransform not in ("smearing", "none"):
            raise ValueError(
                f"retransform must be 'smearing' or 'none', got {retransform!r}"
            )
        self.garch_specs      = GARCHInputs(
            garch_specs if garch_specs is not None else [(garch_model_type, garch_dist)]
        ).specs
        if mode == "residual" and len(self.garch_specs) != 1:
            raise ValueError(
                "mode='residual' needs exactly one GARCH specification: the residual "
                f"is measured against one base forecast; got {len(self.garch_specs)}."
            )
        self.garch_model_type, self.garch_dist = self.garch_specs[0]
        self.garch_p          = garch_p
        self.garch_q          = garch_q
        self.asym_order       = asym_order
        self.mode             = mode
        self.n_lags           = n_lags
        self.use_returns      = use_returns
        self.tune             = validate_tune(tune)
        self.tuning_cache     = tuning_cache
        self.n_trials         = n_trials
        self.optuna_n_jobs    = optuna_n_jobs
        self.xgb_params       = dict(xgb_params or _DEFAULT_XGB_PARAMS)
        self.seed             = seed
        # The log target applies to 'features' mode only: 'residual' mode is fit
        # on proxy - h, which is negative about half the time and has no log.
        self.log_target       = bool(log_target) and mode == "features"
        self.retransform      = retransform
        self.target_floor_q   = target_floor_q
        self._inputs: GARCHInputs | None     = None
        self._xgb: xgb.XGBRegressor | None  = None
        self._params: dict | None            = None
        self._last_sq: np.ndarray | None     = None
        self._last_r:  np.ndarray | None     = None
        self._smearing: float                = 1.0

    @property
    def is_combiner(self) -> bool:
        return len(self.garch_specs) > 1

    @property
    def _garch(self) -> GARCHModel | None:
        """The (first) internal GARCH model — the base of a single-base hybrid."""
        return self._inputs.models[0] if self._inputs is not None else None

    def fit(self, returns: pd.Series, target: pd.Series | None = None) -> "XGBHybridModel":
        """
        `target` is the realized-variance proxy aligned to `returns` — the same
        series the evaluator scores against. None falls back to squared returns.
        """
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        y_raw = resolve_target(r, None if target is None else np.asarray(target, dtype=float))

        self._inputs = GARCHInputs(
            self.garch_specs, p=self.garch_p, q=self.garch_q, asym_order=self.asym_order
        ).fit(returns)
        X, y = _hybrid_design(sq, r, self._inputs.insample_matrix(), y_raw,
                              self.n_lags, self.use_returns, self.mode)
        if self.log_target:
            y = log_variance_target(y, self.target_floor_q)

        params = resolve_hyperparameters(
            self.tune, self.tuning_cache,
            lambda: _optuna_tune(X, y, self.n_trials, self.seed, self.optuna_n_jobs),
            {**self.xgb_params, "random_state": self.seed, "verbosity": 0},
        )
        self._params = dict(params)
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
        Refresh the lagged features AND every GARCH state without refitting.
        The GARCH one-step forecasts are input features, so leaving them stale
        would defeat the point of updating the lags.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < self.n_lags:
            raise ValueError(
                f"update needs at least n_lags={self.n_lags} observations, got {len(r)}."
            )
        if self._inputs is not None:
            self._inputs.update(returns)
        self._last_sq = (r ** 2)[-self.n_lags:].copy()
        self._last_r  = r[-self.n_lags:].copy()
        return self

    def garch_forecasts(self) -> pd.Series:
        """The one-step GARCH forecasts the next prediction will be fed."""
        if self._inputs is None:
            raise RuntimeError("Call .fit() first.")
        return pd.Series(self._inputs.forecasts(), index=self._inputs.names, name="garch_forecast")

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._xgb is None or self._inputs is None:
            raise RuntimeError("Call .fit() first.")
        garch_fc = self._inputs.forecasts()
        row = list(self._last_sq[::-1])
        if self.use_returns:
            row += list(self._last_r[::-1])
        row += list(garch_fc)
        xgb_pred = float(self._xgb.predict(np.array([row]))[0])
        if self.mode == "features":
            result = (
                float(np.exp(xgb_pred)) * self._smearing if self.log_target
                else max(xgb_pred, 1e-10)
            )
        else:
            result = max(float(garch_fc[0]) + xgb_pred, 1e-10)
        return np.full(horizon, result)

    def feature_names(self) -> list[str]:
        names = [f"sq_lag{i + 1}" for i in range(self.n_lags)]
        if self.use_returns:
            names += [f"r_lag{i + 1}" for i in range(self.n_lags)]
        if self.is_combiner:
            names += [f"fc_{t}-{d.upper()}" for t, d in self.garch_specs]
        else:
            names.append("garch_fc")
        return names

    def hyperparameters(self) -> dict:
        """Settings the last fit ran with (tuned or default), for reporting."""
        hp = _xgb_hyperparameters(self)
        hp["garch_inputs"] = len(self.garch_specs)
        return hp

    def __repr__(self) -> str:
        base = (
            f"combiner of {len(self.garch_specs)} GARCH specs" if self.is_combiner
            else f"garch={self.garch_model_type}-{self.garch_dist}"
        )
        return (
            f"XGBHybridModel({base}, mode={self.mode!r}, "
            f"n_lags={self.n_lags}, tune={self.tune!r})"
        )
