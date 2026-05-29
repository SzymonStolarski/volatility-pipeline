from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import numpy as np
import pandas as pd
from .metrics import metrics_summary, compute_loss


@dataclass
class ForecastResult:
    """Out-of-sample forecast results for a single model."""
    name: str
    forecasts: pd.Series          # one-step-ahead conditional variance forecasts
    actuals: pd.Series            # realized variance proxy (squared log-returns)
    refit_indices: list[int] = field(default_factory=list)

    @property
    def errors(self) -> pd.Series:
        """Forecast error: forecast minus actual."""
        return self.forecasts - self.actuals

    def metrics(self) -> dict:
        """Return dict with RMSE, MAE, MSE, QLIKE."""
        return metrics_summary(self.forecasts.values, self.actuals.values)

    def loss_series(self, loss: str = "squared") -> pd.Series:
        """Per-period loss series, used by DM test and MCS."""
        return pd.Series(
            compute_loss(self.forecasts.values, self.actuals.values, loss),
            index=self.forecasts.index,
            name=loss,
        )

    def __repr__(self) -> str:
        m = self.metrics()
        return (
            f"ForecastResult(name={self.name!r}, n={len(self.forecasts)}, "
            f"RMSE={m['RMSE']:.6e})"
        )


class RollingEvaluator:
    """
    Out-of-sample evaluator supporting expanding and sliding windows.

    window_type="expanding" (default): recursive scheme — at step t the model
    uses all returns from the beginning up to t (growing window).

    window_type="sliding": fixed-width scheme — at step t the model uses the
    most recent `len(train_returns)` observations, so the window shifts by one
    at each step while its length stays constant.

    Parameters are re-estimated every `refit_every` steps in both modes.
    """

    def __init__(
        self,
        n_ahead: int = 1,
        refit_every: int = 10,
        window_type: Literal["expanding", "sliding"] = "expanding",
    ) -> None:
        if n_ahead != 1:
            raise NotImplementedError("Only n_ahead=1 is currently supported.")
        if window_type not in ("expanding", "sliding"):
            raise ValueError("window_type must be 'expanding' or 'sliding'.")
        self.n_ahead = n_ahead
        self.refit_every = refit_every
        self.window_type = window_type

    def evaluate(
        self,
        model_factory: Callable,
        name: str,
        train_returns: pd.Series,
        test_returns: pd.Series,
        verbose: bool = False,
    ) -> ForecastResult:
        """
        Run the evaluation loop.

        Parameters
        ----------
        model_factory : callable() -> model with .fit(returns) and .forecast_variance(horizon)
        name          : display label for this model
        train_returns : initial training window (log returns as pd.Series);
                        also sets the fixed window length when window_type="sliding"
        test_returns  : evaluation period (log returns as pd.Series)
        verbose       : print a line each time the model is re-fitted
        """
        all_returns = pd.concat([train_returns, test_returns])
        n_train = len(train_returns)
        n_test = len(test_returns)

        forecasts = np.empty(n_test)
        refit_indices: list[int] = []
        model = None

        for i in range(n_test):
            if i % self.refit_every == 0:
                if self.window_type == "sliding":
                    current_train = all_returns.iloc[i : n_train + i]
                else:
                    current_train = all_returns.iloc[: n_train + i]
                model = model_factory()
                try:
                    model.fit(current_train)
                except Exception as exc:
                    raise RuntimeError(
                        f"[{name}] Fitting failed at test step {i}: {exc}"
                    ) from exc
                refit_indices.append(i)
                if verbose:
                    print(f"  [{name}] re-fitted at step {i}/{n_test} ({self.window_type})")

            fc = model.forecast_variance(horizon=self.n_ahead)
            forecasts[i] = float(fc[0])

        actuals = test_returns.values ** 2  # squared log-returns as variance proxy

        return ForecastResult(
            name=name,
            forecasts=pd.Series(forecasts, index=test_returns.index),
            actuals=pd.Series(actuals, index=test_returns.index),
            refit_indices=refit_indices,
        )

    def evaluate_many(
        self,
        specs: list[tuple[Callable, str]],
        train_returns: pd.Series,
        test_returns: pd.Series,
        verbose: bool = True,
    ) -> dict[str, ForecastResult]:
        """
        Evaluate multiple models in sequence.

        Parameters
        ----------
        specs : list of (factory_callable, name) tuples
        """
        results: dict[str, ForecastResult] = {}
        for factory, name in specs:
            if verbose:
                print(f"Evaluating {name}...")
            results[name] = self.evaluate(
                factory, name, train_returns, test_returns, verbose=verbose
            )
        return results
