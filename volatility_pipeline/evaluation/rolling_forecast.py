from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import multiprocessing
import numpy as np
import pandas as pd
from .metrics import metrics_summary, compute_loss


# ---------------------------------------------------------------------------
# Module-level helper required for pickling in ProcessPoolExecutor.
# Must NOT be a lambda or a nested function.
# ---------------------------------------------------------------------------

def _evaluate_spec(
    evaluator: "RollingEvaluator",
    factory: Callable,
    name: str,
    train_returns: pd.Series,
    test_returns: pd.Series,
    actuals_series: pd.Series | None = None,
) -> tuple[str, "ForecastResult"]:
    """Worker function for parallel evaluation of a single model spec."""
    return name, evaluator.evaluate(
        factory, name, train_returns, test_returns,
        actuals_series=actuals_series, verbose=False,
    )


@dataclass
class ForecastResult:
    """Out-of-sample forecast results for a single model."""
    name: str
    forecasts: pd.Series          # one-step-ahead conditional variance forecasts
    actuals: pd.Series            # realized variance proxy
    refit_indices: list[int] = field(default_factory=list)
    proxy: str = "squared_returns"  # name of the variance proxy used

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
            f"proxy={self.proxy!r}, RMSE={m['RMSE']:.6e})"
        )


class RollingEvaluator:
    """
    Out-of-sample evaluator supporting expanding and sliding windows.

    window_type="expanding" (default): recursive scheme — at step t the model
    uses all returns from the beginning up to t (growing window).

    window_type="sliding": fixed-width scheme — at step t the model uses the
    most recent `window_size` observations, so the window shifts by one at each
    step while its length stays constant. If `window_size` is None it defaults
    to `len(train_returns)`; set it explicitly (e.g. 500 ≈ 2y, 750 ≈ 3y) to run
    a genuinely narrow rolling window that is sensitive to regime shifts. A wide
    sliding window (≈ training length) overlaps almost entirely with the
    expanding window over a short test period and cannot reveal regime effects.

    Parameters are re-estimated every `refit_every` steps in both modes.
    """

    def __init__(
        self,
        n_ahead: int = 1,
        refit_every: int = 10,
        window_type: Literal["expanding", "sliding"] = "expanding",
        window_size: int | None = None,
    ) -> None:
        if n_ahead != 1:
            raise NotImplementedError("Only n_ahead=1 is currently supported.")
        if window_type not in ("expanding", "sliding"):
            raise ValueError("window_type must be 'expanding' or 'sliding'.")
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be a positive integer or None.")
        self.n_ahead = n_ahead
        self.refit_every = refit_every
        self.window_type = window_type
        self.window_size = window_size

    def evaluate(
        self,
        model_factory: Callable,
        name: str,
        train_returns: pd.Series,
        test_returns: pd.Series,
        actuals_series: pd.Series | None = None,
        verbose: bool = False,
    ) -> ForecastResult:
        """
        Run the evaluation loop.

        Parameters
        ----------
        model_factory  : callable() -> model implementing the three-method protocol
                         .fit(returns, target=None), .update(returns) and
                         .forecast_variance(horizon).  `fit` re-estimates
                         parameters; `update` advances the model's state (the
                         GARCH variance recursion, the ML lag window) to the end
                         of the returns it is given WITHOUT re-estimating.
        name           : display label for this model
        train_returns  : initial training window (log returns as pd.Series);
                         also sets the fixed window length when window_type="sliding"
        test_returns   : evaluation period (log returns as pd.Series)
        actuals_series : pre-computed variance proxy (e.g. Garman-Klass or
                         Parkinson).  Reindexed onto test_returns.index for
                         scoring; when it also covers the training period it is
                         passed to fit() as the ML training target so the models
                         learn against the proxy they are scored on.  If None,
                         both fall back to squared log-returns.
        verbose        : print a line each time the model is re-fitted
        """
        all_returns = pd.concat([train_returns, test_returns])
        n_train = len(train_returns)
        n_test = len(test_returns)

        # Training target: the proxy over the whole history, when the caller
        # supplied enough of it. A test-only proxy still scores fine, it just
        # cannot be used for training, so we silently fall back to r^2 there.
        train_target = None
        if actuals_series is not None:
            candidate = actuals_series.reindex(all_returns.index)
            if candidate.notna().all():
                train_target = candidate

        # Sliding-window width: explicit window_size, else full training length.
        width = self.window_size if self.window_size is not None else n_train
        if self.window_type == "sliding" and width > n_train:
            raise ValueError(
                f"window_size ({width}) cannot exceed training length ({n_train})."
            )

        forecasts = np.empty(n_test)
        refit_indices: list[int] = []
        model = None

        fit_start = 0

        for i in range(n_test):
            if i % self.refit_every == 0:
                if self.window_type == "sliding":
                    # Window of `width` obs ending just before the forecast point.
                    # fit_start is held for the whole block, so within a block
                    # the state is filtered from the same origin the parameters
                    # were estimated on (the window grows from `width` to
                    # `width + refit_every - 1` before sliding again).
                    fit_start = n_train + i - width
                else:
                    fit_start = 0
                current_train = all_returns.iloc[fit_start : n_train + i]
                model = model_factory()
                try:
                    model.fit(
                        current_train,
                        target=None if train_target is None
                        else train_target.iloc[fit_start : n_train + i],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"[{name}] Fitting failed at test step {i}: {exc}"
                    ) from exc
                refit_indices.append(i)
                if verbose:
                    print(f"  [{name}] re-fitted at step {i}/{n_test} ({self.window_type})")
            else:
                # Between re-estimations the parameters are held fixed but the
                # state must still advance, otherwise forecast_variance() would
                # return the value computed at the last re-fit and this would
                # not be a one-step-ahead evaluation.
                if not hasattr(model, "update"):
                    raise TypeError(
                        f"[{name}] {type(model).__name__} has no update(returns) method. "
                        f"Models must implement fit/update/forecast_variance; without "
                        f"update the forecast would be frozen between re-fits."
                    )
                try:
                    model.update(all_returns.iloc[fit_start : n_train + i])
                except Exception as exc:
                    raise RuntimeError(
                        f"[{name}] State update failed at test step {i}: {exc}"
                    ) from exc

            fc = model.forecast_variance(horizon=self.n_ahead)
            forecasts[i] = float(fc[0])

        if actuals_series is not None:
            actuals = actuals_series.reindex(test_returns.index).values
            proxy_name = actuals_series.name or "custom"
        else:
            actuals = test_returns.values ** 2
            proxy_name = "squared_returns"

        return ForecastResult(
            name=name,
            forecasts=pd.Series(forecasts, index=test_returns.index),
            actuals=pd.Series(actuals, index=test_returns.index),
            refit_indices=refit_indices,
            proxy=proxy_name,
        )

    def evaluate_many(
        self,
        specs: list[tuple[Callable, str]],
        train_returns: pd.Series,
        test_returns: pd.Series,
        actuals_series: pd.Series | None = None,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> dict[str, ForecastResult]:
        """
        Evaluate multiple models, optionally in parallel.

        Parameters
        ----------
        specs          : list of (factory_callable, name) tuples.
                         For parallel execution (n_jobs != 1) factory callables
                         MUST be picklable — use ``functools.partial`` instead
                         of ``lambda``.
        actuals_series : pre-computed variance proxy (passed through to evaluate())
        n_jobs         : number of parallel worker processes.
                         1  → sequential (default, safe with any factory).
                         -1 → use all available CPU cores.
                         N  → use N worker processes.

        Notes
        -----
        When n_jobs != 1 each model runs in a separate process via
        ``concurrent.futures.ProcessPoolExecutor``.  Results are returned in
        the same order as *specs* regardless of completion order.
        Verbose per-step output is suppressed in workers; a single summary
        line is printed per model once it completes.
        """
        if n_jobs == 1:
            results: dict[str, ForecastResult] = {}
            for factory, name in specs:
                if verbose:
                    print(f"Evaluating {name}...")
                results[name] = self.evaluate(
                    factory, name, train_returns, test_returns,
                    actuals_series=actuals_series, verbose=verbose,
                )
            return results

        # ------------------------------------------------------------------
        # Parallel path
        # ------------------------------------------------------------------
        workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        if verbose:
            print(
                f"Parallel evaluation: {len(specs)} models "
                f"on {workers} worker process(es)..."
            )

        from concurrent.futures import ProcessPoolExecutor, as_completed

        results: dict[str, ForecastResult] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_name = {
                executor.submit(
                    _evaluate_spec, self, factory, name,
                    train_returns, test_returns, actuals_series,
                ): name
                for factory, name in specs
            }
            for future in as_completed(future_to_name):
                name, result = future.result()
                results[name] = result
                if verbose:
                    m = result.metrics()
                    print(f"  [{name}] done  RMSE={m['RMSE']:.4e}")

        # Preserve original spec order
        return {name: results[name] for _, name in specs if name in results}
