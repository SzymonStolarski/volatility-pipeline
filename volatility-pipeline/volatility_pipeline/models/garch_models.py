from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model


# Maps model_type name -> (arch vol name, extra kwargs for arch_model)
_ARCH_VOL: dict[str, tuple[str, dict]] = {
    "GARCH":     ("GARCH",   {"o": 0}),
    "GJR-GARCH": ("GARCH",   {"o": 1}),
    "EGARCH":    ("EGARCH",  {}),
    "APARCH":    ("APARCH",  {}),
    "FIGARCH":   ("FIGARCH", {}),
}

_ARCH_DIST: dict[str, str] = {
    "normal": "normal",
    "t":      "t",
    "ged":    "ged",
}


class GARCHModel:
    """
    Thin wrapper around the arch library covering GARCH, GJR-GARCH,
    EGARCH, APARCH, and FIGARCH with Normal, Student-t, and GED innovations.

    Returns and forecasts are always in original (unscaled) units.
    Internally scales returns by `scale` (default 100) for numerical stability.
    """

    def __init__(
        self,
        model_type: str = "GARCH",
        dist: str = "normal",
        p: int = 1,
        q: int = 1,
        scale: float = 100.0,
    ) -> None:
        if model_type not in _ARCH_VOL:
            raise ValueError(f"model_type must be one of {list(_ARCH_VOL)}, got {model_type!r}")
        if dist not in _ARCH_DIST:
            raise ValueError(f"dist must be one of {list(_ARCH_DIST)}, got {dist!r}")
        self.model_type = model_type
        self.dist = dist
        self.p = p
        self.q = q
        self.scale = scale
        self._result = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series, starting_values=None) -> "GARCHModel":
        scaled = returns * self.scale
        vol_name, extra = _ARCH_VOL[self.model_type]
        am = arch_model(
            scaled,
            vol=vol_name,
            p=self.p,
            q=self.q,
            dist=_ARCH_DIST[self.dist],
            **extra,
        )
        self._result = am.fit(
            disp="off",
            starting_values=starting_values,
            show_warning=False,
        )
        return self

    # ------------------------------------------------------------------
    # In-sample
    # ------------------------------------------------------------------

    def insample_variance(self) -> pd.Series:
        """Fitted conditional variance in original units (not scaled)."""
        self._require_fitted()
        cv = self._result.conditional_volatility  # volatility (std dev), scaled
        return (cv / self.scale) ** 2

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        """
        Multi-step-ahead conditional variance forecasts.

        Returns ndarray of shape (horizon,) in original (unscaled) units.
        Index 0 is the 1-step-ahead forecast.
        """
        self._require_fitted()
        fc = self._result.forecast(horizon=horizon, reindex=False)
        # fc.variance is a DataFrame; last row = most recent forecast window
        return fc.variance.values[-1] / (self.scale ** 2)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def aic(self) -> float:
        self._require_fitted()
        return float(self._result.aic)

    @property
    def bic(self) -> float:
        self._require_fitted()
        return float(self._result.bic)

    @property
    def loglikelihood(self) -> float:
        self._require_fitted()
        return float(self._result.loglikelihood)

    @property
    def params(self) -> pd.Series:
        self._require_fitted()
        return self._result.params

    def summary(self):
        self._require_fitted()
        return self._result.summary()

    def info_criteria(self) -> dict:
        return {"AIC": self.aic, "BIC": self.bic, "LogL": self.loglikelihood}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        return (
            f"GARCHModel(model_type={self.model_type!r}, dist={self.dist!r}, "
            f"p={self.p}, q={self.q})"
        )
