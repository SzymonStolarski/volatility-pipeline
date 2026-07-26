from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model


# Maps model_type name -> (arch vol name, extra kwargs for arch_model)
#
# `o` is the asymmetric (leverage) order. NOTE: arch_model()'s own default is
# o=0, so it must be passed explicitly wherever an asymmetric term is wanted —
# omitting it silently estimates the SYMMETRIC variant.
#   GARCH     o=0 by definition (symmetric).
#   GJR-GARCH o=1 by definition (it *is* GARCH with a leverage indicator).
#   EGARCH    o=1 — the asymmetric log-variance model of Nelson (1991).
#   APARCH    o=1 — the asymmetric power ARCH of Ding et al. (1993).
#   FIGARCH   arch's FIGARCH has no asymmetric term (no `o` argument).
_ARCH_VOL: dict[str, tuple[str, dict]] = {
    "GARCH":     ("GARCH",   {"o": 0}),
    "GJR-GARCH": ("GARCH",   {"o": 1}),
    "EGARCH":    ("EGARCH",  {"o": 1}),
    "APARCH":    ("APARCH",  {"o": 1}),
    "FIGARCH":   ("FIGARCH", {}),
}

# Specifications whose asymmetric order may be overridden via `o`. GARCH and
# GJR-GARCH are excluded because o is what distinguishes them (setting o=1 on
# GARCH would make it a GJR, and o=0 on GJR would make it a plain GARCH), and
# FIGARCH is excluded because it has no asymmetric term at all.
_ASYM_CONFIGURABLE: tuple[str, ...] = ("EGARCH", "APARCH")

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
        o: int | None = None,
    ) -> None:
        if model_type not in _ARCH_VOL:
            raise ValueError(f"model_type must be one of {list(_ARCH_VOL)}, got {model_type!r}")
        if dist not in _ARCH_DIST:
            raise ValueError(f"dist must be one of {list(_ARCH_DIST)}, got {dist!r}")
        if o is not None:
            if model_type not in _ASYM_CONFIGURABLE:
                raise ValueError(
                    f"o is not configurable for {model_type!r}; it is only "
                    f"overridable for {list(_ASYM_CONFIGURABLE)}. "
                    f"(GARCH/GJR-GARCH are defined by o=0/o=1, and FIGARCH has "
                    f"no asymmetric term.)"
                )
            if o not in (0, 1):
                raise ValueError(f"o must be 0 or 1, got {o!r}")
        self.model_type = model_type
        self.dist = dist
        self.p = p
        self.q = q
        self.scale = scale
        self.o = o
        self._result = None

    @property
    def asym_order(self) -> int:
        """
        Asymmetric (leverage) order actually used: the explicit override if one
        was given, otherwise the specification's default. 0 means the fitted
        model is symmetric (no gamma coefficient).
        """
        if self.o is not None:
            return self.o
        return _ARCH_VOL[self.model_type][1].get("o", 0)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series, starting_values=None) -> "GARCHModel":
        scaled = returns * self.scale
        vol_name, defaults = _ARCH_VOL[self.model_type]
        extra = dict(defaults)
        if self.o is not None:
            extra["o"] = self.o
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

    @property
    def std_err(self) -> pd.Series:
        """Standard errors of the estimated coefficients."""
        self._require_fitted()
        return self._result.std_err

    @property
    def tvalues(self) -> pd.Series:
        """t-statistics of the estimated coefficients."""
        self._require_fitted()
        return self._result.tvalues

    @property
    def pvalues(self) -> pd.Series:
        """p-values of the estimated coefficients."""
        self._require_fitted()
        return self._result.pvalues

    def param_table(self) -> pd.DataFrame:
        """
        Coefficient, standard error, t-statistic and p-value per parameter.

        Coefficients refer to returns scaled by `self.scale` (percent by
        default) — see `volatility_pipeline.models.equations.scale_note`.
        """
        self._require_fitted()
        return pd.DataFrame({
            "coef": self._result.params,
            "std_err": self._result.std_err,
            "t_stat": self._result.tvalues,
            "p_value": self._result.pvalues,
        })

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
            f"p={self.p}, q={self.q}, o={self.asym_order})"
        )


def make_garch(
    model_type: str,
    dist: str = "normal",
    asym_order: int | None = None,
    **kwargs,
) -> GARCHModel:
    """
    Build a GARCHModel, applying `asym_order` only where it is meaningful.

    This is the safe way to thread a single notebook-level asymmetry switch
    through a heterogeneous list of specifications: `asym_order` is forwarded
    only for the specifications whose leverage order is configurable
    (EGARCH, APARCH) and silently ignored for GARCH, GJR-GARCH and FIGARCH,
    whose asymmetric order is part of their definition. Passing `asym_order`
    straight to GARCHModel for those would raise.

    Defined at module level (not a lambda or closure) so that
    `functools.partial(make_garch, ...)` stays picklable for
    RollingEvaluator's process-parallel path.

    Parameters
    ----------
    model_type : 'GARCH' | 'GJR-GARCH' | 'EGARCH' | 'APARCH' | 'FIGARCH'
    dist       : 'normal' | 't' | 'ged'
    asym_order : 0 or 1 to override the leverage order of EGARCH/APARCH;
                 None uses each specification's default (asymmetric for those two).
    """
    if asym_order is not None and model_type in _ASYM_CONFIGURABLE:
        kwargs["o"] = asym_order
    return GARCHModel(model_type, dist, **kwargs)
