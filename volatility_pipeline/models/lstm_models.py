from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

from .garch_models import GARCHModel
from .tuning import TuningCache, chrono_split, resolve_hyperparameters, validate_tune
from .targets import (
    _EPS,
    log_variance_target,
    resolve_target,
    smearing_factor,
    winsor_bounds,
    winsorize,
)


def _select_device(device: str | None = None) -> torch.device:
    """Pick the best available compute device unless one is explicitly given."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fit_smearing(
    net: "_LSTMNet",
    X: np.ndarray,
    y: np.ndarray,
    target_scaler: RobustScaler,
    device: torch.device,
) -> float:
    """Smearing factor for a log-scale target, from the in-sample residuals."""
    if len(X) == 0:
        return 1.0
    inv = target_scaler.inverse_transform
    log_pred = inv(_predict_batch(net, X, device).reshape(-1, 1)).ravel()
    log_true = inv(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()
    return smearing_factor(log_true, log_pred)


def _build_sequences(
    features: np.ndarray, target: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    features: (n, n_features) scaled inputs
    target:   (n,) scaled targets

    Returns X of shape (n_samples, lookback, n_features) and y of shape
    (n_samples,), where X[k] = features[i-lookback+1 : i+1] and
    y[k] = target[i+1] for i in range(lookback, n-1).
    """
    n = len(target)
    rows, targets = [], []
    for i in range(lookback, n - 1):
        rows.append(features[i - lookback + 1 : i + 1])
        targets.append(target[i + 1])
    return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.float32)


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def _fit_network(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> _LSTMNet:
    """Train an _LSTMNet with early stopping on a time-ordered validation tail."""
    # Single-threaded CPU training, for two reasons:
    # 1. Safety — sklearn and torch pip wheels each bundle their own libomp on
    #    macOS; when both are loaded (this package imports both), torch entering
    #    a multi-threaded CPU parallel region segfaults. One thread avoids the
    #    OpenMP parallel region entirely, regardless of import order.
    # 2. Speed — nets this small gain nothing from >1 threads (measured parity),
    #    and one thread per process is required anyway when models run under
    #    process-level parallelism (RollingEvaluator n_jobs != 1).
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)

    n = len(X)
    # Early-stopping tail: the LAST val_fraction of whatever the net is given.
    n_train = chrono_split(n, val_frac=val_fraction)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    X_train, y_train = X_t[:n_train], y_t[:n_train]
    X_val, y_val = X_t[n_train:], y_t[n_train:]

    net = _LSTMNet(n_features, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state, best_val, no_improve = None, float("inf"), 0

    for _ in range(max_epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val.to(device)), y_val.to(device)).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _predict_one(net: _LSTMNet, x: np.ndarray, device: torch.device) -> float:
    net.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x[np.newaxis, ...]).to(device)
        return float(net(x_t).cpu().item())


def _predict_batch(
    net: _LSTMNet, X: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    """In-sample predictions over the training sequences (used for smearing)."""
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start: start + batch_size]).to(device)
            out.append(net(xb).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.empty(0, dtype=float)


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

#: Search space for the LSTM. Deliberately small: every trial trains a full
#: network, so this is 30-60x more expensive per trial than the XGBoost search,
#: and a wide space would buy noise rather than fit.
#:
#: The LOOKBACK IS NOT SEARCHED, and must not be. It is the number of lagged
#: returns and lagged squared returns each prediction sees (every timestep
#: carries [r, r^2]), i.e. the model's INFORMATION SET, not a capacity or
#: optimisation setting. The frozen specification gives both ML families the
#: same one — 10 lagged returns and 10 lagged squared returns — as the condition
#: for a fair comparison of model classes; XGBoost's n_lags is fixed for the
#: same reason and is not in its search either. Searching it would let the LSTM
#: pick a longer history than XGBoost on validation data. `_optuna_tune_lstm`
#: rejects a search space that contains it.
#:
#: hidden_size capacity. With ~3,000 daily observations anything past 64 has far
#:             more parameters than the data can identify.
#: num_layers  depth. Note that dropout between layers only exists when this is
#:             greater than 1, so the two interact.
#: dropout     regularisation.
#: lr          matters mostly through whether the net converges inside
#:             max_epochs at all, rather than through the optimum it reaches.
#: batch_size  gradient noise and wall-clock, jointly.
#:
#: max_epochs, patience and val_fraction are deliberately NOT searched: they are
#: the training budget, and letting trials differ in budget would confound
#: "better architecture" with "trained longer".
LSTM_SEARCH_SPACE: dict = {
    "hidden_size": [16, 32, 64],
    "num_layers":  [1, 2],
    "dropout":     (0.0, 0.4),
    "lr":          (1e-4, 1e-2),
    "batch_size":  [32, 64, 128],
}

#: Hyperparameters the search may set. Everything else is fixed by construction.
LSTM_TUNABLE: tuple[str, ...] = tuple(LSTM_SEARCH_SPACE)

#: Fixed by the specification, never tuned: they define what the model sees
#: (lookback) or how long it trains (the budget). See the note above.
LSTM_NOT_TUNABLE: tuple[str, ...] = ("lookback", "max_epochs", "patience", "val_fraction")


def _optuna_tune_lstm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int,
    seed: int,
    max_epochs: int,
    patience: int,
    val_fraction: float,
    device: torch.device,
    search_space: dict | None = None,
    n_jobs: int = 1,
) -> dict:
    """
    Tune the network on a chronological hold-out carved from the training
    window, mirroring what `_optuna_tune` does for XGBoost.

    The split is NESTED, and has to be. `_fit_network` already holds out the
    last `val_fraction` of whatever it is given for early stopping; scoring
    trials on that same tail would select hyperparameters on the data used to
    decide when to stop training, which flatters every trial that happened to
    stop at a lucky epoch. So the training window is cut 80/20 in time, trials
    train on the first 80% (inside which early stopping takes its own tail) and
    are scored on the last 20%, which no trial has trained on or stopped on.
    The winning configuration is then refitted on the whole window by the
    caller, exactly as the XGBoost path does.

    `X, y` are the training sequences, built by the caller at the model's
    FIXED lookback (the hybrid builds them differently: it carries the GARCH
    forecast as an extra channel). The lookback is not searched, so the
    sequences are the same for every trial.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = search_space or LSTM_SEARCH_SPACE
    forbidden = sorted(set(space) & set(LSTM_NOT_TUNABLE))
    if forbidden:
        raise ValueError(
            f"{forbidden} cannot be searched: the lookback is the number of "
            f"lagged returns and lagged squared returns the model sees (its "
            f"information set, fixed by the specification and shared with "
            f"XGBoost's n_lags), and max_epochs/patience/val_fraction are the "
            f"training budget. Set them on the model instead."
        )

    split = chrono_split(len(X), train_frac=0.8)   # earlier 80% trains, later 20% scores
    if split < 30 or len(X) - split < 10:
        raise ValueError(
            f"too little history to tune: {len(X)} training sequences leave no "
            f"usable chronological hold-out."
        )
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    def objective(trial) -> float:
        hp = {
            "hidden_size": trial.suggest_categorical("hidden_size", space["hidden_size"]),
            "num_layers":  trial.suggest_categorical("num_layers", space["num_layers"]),
            "dropout":     trial.suggest_float("dropout", *space["dropout"]),
            "lr":          trial.suggest_float("lr", *space["lr"], log=True),
            "batch_size":  trial.suggest_categorical("batch_size", space["batch_size"]),
        }
        net = _fit_network(
            X_tr, y_tr, n_features=X.shape[-1],
            hidden_size=hp["hidden_size"], num_layers=hp["num_layers"],
            dropout=hp["dropout"], lr=hp["lr"], max_epochs=max_epochs,
            patience=patience, batch_size=hp["batch_size"],
            val_fraction=val_fraction, seed=seed, device=device,
        )
        pred = _predict_batch(net, X_val, device)
        return float(np.mean((pred - y_val) ** 2))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    return dict(study.best_params)


class _TunableLSTM:
    """
    Shared tuning plumbing for the standalone and hybrid LSTM models.

    The resolved hyperparameters (LSTM_TUNABLE) are written back onto the
    instance in fit(), so every later use of self.hidden_size and the rest picks
    them up without any further threading. Constructor values of those are
    therefore the DEFAULTS and the starting point, not necessarily the values a
    fitted model ran with; read them back off the fitted instance, or off
    `.tuning_cache`. The lookback is never among them: it is the information set
    (lagged returns and lagged squared returns) and stays at the constructor
    value whatever the tuning cadence.

    Mutation does not leak between re-estimations: RollingEvaluator builds a
    fresh model each time, and under tune="first" the shared cache hands every
    one of them the same values.
    """

    def _target_floor_q(self) -> float:
        q = getattr(self, "target_floor_q", None)
        return self.winsor_limits[0] if q is None else q

    def _default_hp(self) -> dict:
        return {k: getattr(self, k) for k in LSTM_TUNABLE}

    def hyperparameters(self) -> dict:
        """Settings the last fit ran with (tuned or default), for reporting:
        the searched ones, plus the fixed information set and training budget."""
        return {k: getattr(self, k) for k in (*LSTM_TUNABLE, *LSTM_NOT_TUNABLE)}

    def _apply_hp(self, hp: dict) -> None:
        for k, v in hp.items():
            if k in LSTM_TUNABLE:
                setattr(self, k, v)

    def _resolve_hp(self, seq_builder) -> None:
        if self.tune == "always":
            warnings.warn(
                f"{type(self).__name__} with tune='always' runs a full "
                f"{self.n_trials}-trial network search at EVERY re-estimation. "
                f"Over a ~1000-day test period with refit_every=10 that is ~101 "
                f"searches, which is hours per model. tune='first' is the "
                f"symmetric counterpart to the XGBoost setting and is what the "
                f"comparison actually needs.",
                RuntimeWarning,
                stacklevel=3,
            )
        hp = resolve_hyperparameters(
            self.tune,
            self.tuning_cache,
            lambda: _optuna_tune_lstm(
                *seq_builder(self.lookback),
                n_trials=self.n_trials, seed=self.seed,
                max_epochs=self.max_epochs, patience=self.patience,
                val_fraction=self.val_fraction, device=self.device,
            ),
            self._default_hp(),
        )
        self._apply_hp(hp)


class LSTMVolatilityModel(_TunableLSTM):
    """
    Standalone LSTM volatility forecaster. Compatible with RollingEvaluator
    (.fit / .forecast_variance interface).

    Inputs at each timestep: [return, squared_return], winsorized and
    RobustScaler-scaled (fit on the training window only, refit every call).
    Target: log(squared_return) one step ahead (lower tail floored),
    RobustScaler-scaled.
    forecast_variance() inverse-scales, exponentiates and applies the
    retransformation correction to return a forecast of E[r^2] in variance
    units, comparable with the GARCH and XGB forecasts.

    Parameters
    ----------
    retransform : how to map the log-scale network output back to variance units
        'smearing' — Duan (1983) smearing factor estimated on the fit residuals
                     (default; required for the forecast to estimate E[r^2|X]).
        'none'     — plain exp(), which estimates the conditional GEOMETRIC mean
                     and is biased low by roughly 5x on daily returns. Kept only
                     to reproduce pre-fix results.
    """

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        val_fraction: float = 0.15,
        tune: str = "never",
        tuning_cache: TuningCache | None = None,
        n_trials: int = 20,
        winsor_limits: tuple[float, float] = (0.01, 0.01),
        target_floor_q: float | None = None,
        retransform: str = "smearing",
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if retransform not in ("smearing", "none"):
            raise ValueError(
                f"retransform must be 'smearing' or 'none', got {retransform!r}"
            )
        self.retransform   = retransform
        self.lookback      = lookback
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.lr            = lr
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.batch_size    = batch_size
        self.val_fraction  = val_fraction
        self.tune          = validate_tune(tune)
        self.tuning_cache  = tuning_cache
        self.n_trials      = n_trials
        self.winsor_limits = winsor_limits
        # Lower-tail floor of the log target. None keeps the historical
        # behaviour (winsor_limits[0]); 0.0 defers to a floor already applied
        # to the target series at data preparation (prepare_series), which
        # makes this one a no-op.
        self.target_floor_q = target_floor_q
        self.seed          = seed
        self.device        = _select_device(device)

        self._net: _LSTMNet | None = None
        self._feature_scaler: RobustScaler | None = None
        self._target_scaler:  RobustScaler | None = None
        self._last_window:    np.ndarray | None = None
        self._smearing:       float = 1.0
        # Winsorisation bounds are frozen at fit time so that update() applies
        # the same transform the network was trained under.
        self._r_bounds:  tuple[float, float] | None = None
        self._sq_bounds: tuple[float, float] | None = None

    def _scaled_features(self, r: np.ndarray) -> np.ndarray:
        r_w  = winsorize(r, self._r_bounds)
        sq_w = winsorize(r ** 2, self._sq_bounds)
        return self._feature_scaler.transform(np.column_stack([r_w, sq_w]))

    def fit(self, returns: pd.Series, target: pd.Series | None = None) -> "LSTMVolatilityModel":
        """
        `target` is the realized-variance proxy aligned to `returns` — the same
        series the evaluator scores against. None falls back to squared returns.
        """
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        y_raw = resolve_target(r, None if target is None else np.asarray(target, dtype=float))

        self._r_bounds  = winsor_bounds(r, self.winsor_limits)
        self._sq_bounds = winsor_bounds(sq, self.winsor_limits)

        features = np.column_stack([
            winsorize(r, self._r_bounds), winsorize(sq, self._sq_bounds)
        ])
        self._feature_scaler = RobustScaler().fit(features)
        features_scaled = self._feature_scaler.transform(features)

        log_var = log_variance_target(y_raw, self._target_floor_q())
        self._target_scaler = RobustScaler().fit(log_var.reshape(-1, 1))
        target_scaled = self._target_scaler.transform(log_var.reshape(-1, 1)).ravel()

        def _seq(lookback):
            return _build_sequences(features_scaled, target_scaled, lookback)

        # May run the search and overwrite hidden_size / num_layers / ... (never lookback)
        self._resolve_hp(_seq)

        X, y = _seq(self.lookback)
        self._net = _fit_network(
            X, y, n_features=X.shape[-1],
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size,
            val_fraction=self.val_fraction, seed=self.seed, device=self.device,
        )
        self._smearing = (
            1.0 if self.retransform == "none"
            else _fit_smearing(self._net, X, y, self._target_scaler, self.device)
        )
        self._last_window = features_scaled[-self.lookback:].astype(np.float32)
        return self

    def update(self, returns: pd.Series) -> "LSTMVolatilityModel":
        """
        Slide the input window forward without retraining.

        The winsorisation bounds and both scalers stay exactly as fitted —
        refitting them on the growing sample would change the transform the
        network was trained under.
        """
        if self._net is None:
            raise RuntimeError("Call .fit() first.")
        r = np.asarray(returns, dtype=float)
        if len(r) < self.lookback:
            raise ValueError(
                f"update needs at least lookback={self.lookback} observations, got {len(r)}."
            )
        self._last_window = self._scaled_features(r)[-self.lookback:].astype(np.float32)
        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call .fit() first.")
        pred_scaled = _predict_one(self._net, self._last_window, self.device)
        log_pred = self._target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        pred = max(float(np.exp(log_pred)) * self._smearing, 1e-10)
        return np.full(horizon, pred)

    def feature_names(self) -> list[str]:
        return ["return", "sq_return"]

    def __repr__(self) -> str:
        return (
            f"LSTMVolatilityModel(lookback={self.lookback}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, retransform={self.retransform!r}, "
            f"device={self.device.type!r})"
        )


class LSTMHybridModel(_TunableLSTM):
    """
    Hybrid LSTM + GARCH volatility model. Compatible with RollingEvaluator.

    mode='features':
        Inputs at each timestep are [return, squared_return] plus the GARCH
        one-step-ahead forecast h_{t+1|t} (log-scaled), broadcast across every
        timestep of the lookback window. Final forecast = LSTM output
        (inverse-scaled, exponentiated, retransformation-corrected).

    mode='residual':
        Inputs are [return, squared_return] only (as in the standalone model).
        Target is the GARCH residual: sq[t+1] - h_{t+1|t} (RobustScaler, no log
        transform since residuals can be negative).
        Final forecast = GARCH_forecast + LSTM_residual_correction.

    `retransform` applies to 'features' mode only — see LSTMVolatilityModel.
    'residual' mode is fit on the level scale and needs no correction.

    The internal GARCH model is re-estimated on every .fit() call, so the
    hybrid model is self-contained and works transparently with
    RollingEvaluator's expanding/sliding window refitting logic.
    """

    def __init__(
        self,
        garch_model_type: str = "GARCH",
        garch_dist: str = "normal",
        garch_p: int = 1,
        garch_q: int = 1,
        mode: str = "features",
        lookback: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        val_fraction: float = 0.15,
        tune: str = "never",
        tuning_cache: TuningCache | None = None,
        n_trials: int = 20,
        winsor_limits: tuple[float, float] = (0.01, 0.01),
        target_floor_q: float | None = None,
        retransform: str = "smearing",
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if mode not in ("features", "residual"):
            raise ValueError(f"mode must be 'features' or 'residual', got {mode!r}")
        if retransform not in ("smearing", "none"):
            raise ValueError(
                f"retransform must be 'smearing' or 'none', got {retransform!r}"
            )
        self.retransform      = retransform
        self.garch_model_type = garch_model_type
        self.garch_dist       = garch_dist
        self.garch_p          = garch_p
        self.garch_q          = garch_q
        self.mode             = mode
        self.lookback         = lookback
        self.hidden_size      = hidden_size
        self.num_layers       = num_layers
        self.dropout          = dropout
        self.lr               = lr
        self.max_epochs       = max_epochs
        self.patience         = patience
        self.batch_size       = batch_size
        self.val_fraction     = val_fraction
        self.tune             = validate_tune(tune)
        self.tuning_cache     = tuning_cache
        self.n_trials         = n_trials
        self.winsor_limits    = winsor_limits
        self.target_floor_q   = target_floor_q   # see LSTMVolatilityModel
        self.seed             = seed
        self.device           = _select_device(device)

        self._garch: GARCHModel | None = None
        self._net: _LSTMNet | None = None
        self._feature_scaler: RobustScaler | None = None
        self._garch_scaler:   RobustScaler | None = None  # 'features' mode only
        self._target_scaler:  RobustScaler | None = None  # log target ('features') or residual ('residual')
        self._last_window:    np.ndarray | None = None
        self._smearing:       float = 1.0  # 'features' mode only
        # Frozen at fit time so that update() applies the same transform the
        # network was trained under.
        self._r_bounds:  tuple[float, float] | None = None
        self._sq_bounds: tuple[float, float] | None = None

    def _scaled_features(self, r: np.ndarray) -> np.ndarray:
        r_w  = winsorize(r, self._r_bounds)
        sq_w = winsorize(r ** 2, self._sq_bounds)
        return self._feature_scaler.transform(np.column_stack([r_w, sq_w]))

    def _current_window(self, base_scaled: np.ndarray) -> np.ndarray:
        """Input window ending at the last observation, with the GARCH channel."""
        if self.mode != "features":
            return base_scaled[-self.lookback:].astype(np.float32)
        garch_fc_now = float(self._garch.forecast_variance(horizon=1)[0])
        log_g_now = np.log(garch_fc_now + _EPS)
        g_now_scaled = self._garch_scaler.transform([[log_g_now]])[0, 0]
        g_col = np.full((self.lookback, 1), g_now_scaled, dtype=np.float32)
        return np.hstack([base_scaled[-self.lookback:], g_col]).astype(np.float32)

    def fit(self, returns: pd.Series, target: pd.Series | None = None) -> "LSTMHybridModel":
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
        g_var = self._garch.insample_variance().values  # h_{t|t-1}

        self._r_bounds  = winsor_bounds(r, self.winsor_limits)
        self._sq_bounds = winsor_bounds(sq, self.winsor_limits)

        base_features = np.column_stack([
            winsorize(r, self._r_bounds), winsorize(sq, self._sq_bounds)
        ])
        self._feature_scaler = RobustScaler().fit(base_features)
        base_scaled = self._feature_scaler.transform(base_features)

        if self.mode == "features":
            log_g = np.log(g_var + _EPS)
            self._garch_scaler = RobustScaler().fit(log_g.reshape(-1, 1))
            g_scaled = self._garch_scaler.transform(log_g.reshape(-1, 1)).ravel()

            log_var = log_variance_target(y_raw, self._target_floor_q())
            self._target_scaler = RobustScaler().fit(log_var.reshape(-1, 1))
            target_scaled = self._target_scaler.transform(log_var.reshape(-1, 1)).ravel()

            def _seq(lookback):
                rows, targets = [], []
                for i in range(lookback, n - 1):
                    seq = base_scaled[i - lookback + 1 : i + 1]
                    g_col = np.full((lookback, 1), g_scaled[i + 1], dtype=np.float32)
                    rows.append(np.hstack([seq, g_col]))
                    targets.append(target_scaled[i + 1])
                return (np.array(rows, dtype=np.float32),
                        np.array(targets, dtype=np.float32))
        else:  # residual
            residual = y_raw - g_var
            self._target_scaler = RobustScaler().fit(residual.reshape(-1, 1))
            target_scaled = self._target_scaler.transform(residual.reshape(-1, 1)).ravel()

            def _seq(lookback):
                return _build_sequences(base_scaled, target_scaled, lookback)

        # May run the search and overwrite hidden_size / num_layers / ... (never lookback)
        self._resolve_hp(_seq)

        X, y = _seq(self.lookback)
        self._net = _fit_network(
            X, y, n_features=X.shape[-1],
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size,
            val_fraction=self.val_fraction, seed=self.seed, device=self.device,
        )

        # Retransformation correction applies only to the log-scale target used
        # by 'features' mode; 'residual' mode is fit on the level scale.
        self._smearing = (
            _fit_smearing(self._net, X, y, self._target_scaler, self.device)
            if (self.mode == "features" and self.retransform != "none")
            else 1.0
        )

        self._last_window = self._current_window(base_scaled)
        return self

    def update(self, returns: pd.Series) -> "LSTMHybridModel":
        """
        Slide the input window forward and roll the GARCH state, without
        retraining either. Scalers and winsorisation bounds stay as fitted.
        """
        if self._net is None or self._garch is None:
            raise RuntimeError("Call .fit() first.")
        r = np.asarray(returns, dtype=float)
        if len(r) < self.lookback:
            raise ValueError(
                f"update needs at least lookback={self.lookback} observations, got {len(r)}."
            )
        self._garch.update(returns)
        self._last_window = self._current_window(self._scaled_features(r))
        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._net is None or self._garch is None:
            raise RuntimeError("Call .fit() first.")
        pred_scaled = _predict_one(self._net, self._last_window, self.device)

        if self.mode == "features":
            log_pred = self._target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            result = max(float(np.exp(log_pred)) * self._smearing, 1e-10)
        else:
            garch_fc = float(self._garch.forecast_variance(horizon=1)[0])
            residual = self._target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            result = max(garch_fc + float(residual), 1e-10)

        return np.full(horizon, result)

    def feature_names(self) -> list[str]:
        names = ["return", "sq_return"]
        if self.mode == "features":
            names.append("garch_fc")
        return names

    def __repr__(self) -> str:
        return (
            f"LSTMHybridModel(garch={self.garch_model_type}-{self.garch_dist}, "
            f"mode={self.mode!r}, lookback={self.lookback}, "
            f"retransform={self.retransform!r}, device={self.device.type!r})"
        )
