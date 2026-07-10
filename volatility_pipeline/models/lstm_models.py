from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

from .garch_models import GARCHModel


_EPS = 1e-12


def _select_device(device: str | None = None) -> torch.device:
    """Pick the best available compute device unless one is explicitly given."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _winsor_bounds(x: np.ndarray, limits: tuple[float, float]) -> tuple[float, float]:
    lo_q, hi_q = limits
    lo, hi = np.quantile(x, [lo_q, 1.0 - hi_q])
    return float(lo), float(hi)


def _winsorize(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(x, bounds[0], bounds[1])


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
    torch.manual_seed(seed)

    n = len(X)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val

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


class LSTMVolatilityModel:
    """
    Standalone LSTM volatility forecaster. Compatible with RollingEvaluator
    (.fit / .forecast_variance interface).

    Inputs at each timestep: [return, squared_return], winsorized and
    RobustScaler-scaled (fit on the training window only, refit every call).
    Target: log(squared_return) one step ahead, RobustScaler-scaled.
    forecast_variance() inverse-scales and exponentiates back to variance units.
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
        winsor_limits: tuple[float, float] = (0.01, 0.01),
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.lookback      = lookback
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.lr            = lr
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.batch_size    = batch_size
        self.val_fraction  = val_fraction
        self.winsor_limits = winsor_limits
        self.seed          = seed
        self.device        = _select_device(device)

        self._net: _LSTMNet | None = None
        self._feature_scaler: RobustScaler | None = None
        self._target_scaler:  RobustScaler | None = None
        self._last_window:    np.ndarray | None = None

    def fit(self, returns: pd.Series) -> "LSTMVolatilityModel":
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2

        r_w  = _winsorize(r, _winsor_bounds(r, self.winsor_limits))
        sq_w = _winsorize(sq, _winsor_bounds(sq, self.winsor_limits))

        features = np.column_stack([r_w, sq_w])
        self._feature_scaler = RobustScaler().fit(features)
        features_scaled = self._feature_scaler.transform(features)

        log_sq = np.log(sq + _EPS)
        self._target_scaler = RobustScaler().fit(log_sq.reshape(-1, 1))
        target_scaled = self._target_scaler.transform(log_sq.reshape(-1, 1)).ravel()

        X, y = _build_sequences(features_scaled, target_scaled, self.lookback)
        self._net = _fit_network(
            X, y, n_features=X.shape[-1],
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size,
            val_fraction=self.val_fraction, seed=self.seed, device=self.device,
        )
        self._last_window = features_scaled[-self.lookback:].astype(np.float32)
        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call .fit() first.")
        pred_scaled = _predict_one(self._net, self._last_window, self.device)
        log_pred = self._target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        pred = max(float(np.exp(log_pred)), 1e-10)
        return np.full(horizon, pred)

    def feature_names(self) -> list[str]:
        return ["return", "sq_return"]

    def __repr__(self) -> str:
        return (
            f"LSTMVolatilityModel(lookback={self.lookback}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, device={self.device.type!r})"
        )


class LSTMHybridModel:
    """
    Hybrid LSTM + GARCH volatility model. Compatible with RollingEvaluator.

    mode='features':
        Inputs at each timestep are [return, squared_return] plus the GARCH
        one-step-ahead forecast h_{t+1|t} (log-scaled), broadcast across every
        timestep of the lookback window. Final forecast = LSTM output
        (inverse-scaled, exponentiated).

    mode='residual':
        Inputs are [return, squared_return] only (as in the standalone model).
        Target is the GARCH residual: sq[t+1] - h_{t+1|t} (RobustScaler, no log
        transform since residuals can be negative).
        Final forecast = GARCH_forecast + LSTM_residual_correction.

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
        winsor_limits: tuple[float, float] = (0.01, 0.01),
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if mode not in ("features", "residual"):
            raise ValueError(f"mode must be 'features' or 'residual', got {mode!r}")
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
        self.winsor_limits    = winsor_limits
        self.seed             = seed
        self.device           = _select_device(device)

        self._garch: GARCHModel | None = None
        self._net: _LSTMNet | None = None
        self._feature_scaler: RobustScaler | None = None
        self._garch_scaler:   RobustScaler | None = None  # 'features' mode only
        self._target_scaler:  RobustScaler | None = None  # log target ('features') or residual ('residual')
        self._last_window:    np.ndarray | None = None

    def fit(self, returns: pd.Series) -> "LSTMHybridModel":
        r  = np.asarray(returns, dtype=float)
        sq = r ** 2
        n  = len(r)

        self._garch = GARCHModel(
            self.garch_model_type, self.garch_dist, self.garch_p, self.garch_q
        )
        self._garch.fit(returns)
        g_var = self._garch.insample_variance().values  # h_{t|t-1}

        r_w  = _winsorize(r, _winsor_bounds(r, self.winsor_limits))
        sq_w = _winsorize(sq, _winsor_bounds(sq, self.winsor_limits))

        base_features = np.column_stack([r_w, sq_w])
        self._feature_scaler = RobustScaler().fit(base_features)
        base_scaled = self._feature_scaler.transform(base_features)

        if self.mode == "features":
            log_g = np.log(g_var + _EPS)
            self._garch_scaler = RobustScaler().fit(log_g.reshape(-1, 1))
            g_scaled = self._garch_scaler.transform(log_g.reshape(-1, 1)).ravel()

            log_sq = np.log(sq + _EPS)
            self._target_scaler = RobustScaler().fit(log_sq.reshape(-1, 1))
            target_scaled = self._target_scaler.transform(log_sq.reshape(-1, 1)).ravel()

            rows, targets = [], []
            for i in range(self.lookback, n - 1):
                seq = base_scaled[i - self.lookback + 1 : i + 1]
                g_col = np.full((self.lookback, 1), g_scaled[i + 1], dtype=np.float32)
                rows.append(np.hstack([seq, g_col]))
                targets.append(target_scaled[i + 1])
            X = np.array(rows, dtype=np.float32)
            y = np.array(targets, dtype=np.float32)
        else:  # residual
            residual = sq - g_var
            self._target_scaler = RobustScaler().fit(residual.reshape(-1, 1))
            target_scaled = self._target_scaler.transform(residual.reshape(-1, 1)).ravel()
            X, y = _build_sequences(base_scaled, target_scaled, self.lookback)

        self._net = _fit_network(
            X, y, n_features=X.shape[-1],
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size,
            val_fraction=self.val_fraction, seed=self.seed, device=self.device,
        )

        if self.mode == "features":
            garch_fc_now = float(self._garch.forecast_variance(horizon=1)[0])
            log_g_now = np.log(garch_fc_now + _EPS)
            g_now_scaled = self._garch_scaler.transform([[log_g_now]])[0, 0]
            base_window = base_scaled[-self.lookback:]
            g_col = np.full((self.lookback, 1), g_now_scaled, dtype=np.float32)
            self._last_window = np.hstack([base_window, g_col]).astype(np.float32)
        else:
            self._last_window = base_scaled[-self.lookback:].astype(np.float32)

        return self

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._net is None or self._garch is None:
            raise RuntimeError("Call .fit() first.")
        pred_scaled = _predict_one(self._net, self._last_window, self.device)

        if self.mode == "features":
            log_pred = self._target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            result = max(float(np.exp(log_pred)), 1e-10)
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
            f"mode={self.mode!r}, lookback={self.lookback}, device={self.device.type!r})"
        )
