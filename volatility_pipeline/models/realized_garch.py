"""
Realized GARCH (Hansen, Huang & Shek 2012), log-linear RealGARCH(1,1).

WHY THIS MODEL IS HERE
----------------------
Every GARCH specification in this project is estimated by QMLE on close-to-close
returns, so the only thing it ever learns about day t's volatility is r_t. The
ML models, by contrast, are trained on a range-based proxy and therefore see the
daily HIGH-LOW range as well. When the ML models win, part of the gap is the
model class and part is simply the richer information set, and a comparison
between the two cannot separate them.

Realized GARCH closes that gap from the other side: it is a GARCH model that
also observes the realized measure. Comparing it with the plain GARCH family
isolates the value of the range information within one model class, and
comparing it with the ML models isolates the value of the model class at a
common information set. That is the comparison this file exists to make
possible; without it "ML beats GARCH" is not an interpretable statement.

THE MODEL
---------
    return equation       r_t    = sqrt(h_t) * z_t,        z_t ~ iid N(0, 1)
    GARCH equation        log h_t = omega + beta*log h_{t-1} + gamma*log x_{t-1}
    measurement equation  log x_t = xi + phi*log h_t + tau(z_t) + u_t,
                                                        u_t ~ iid N(0, sigma_u^2)
    leverage function     tau(z)  = tau1*z + tau2*(z^2 - 1)

x_t is the realized measure for day t. The measurement equation is what makes
this more than a GARCH with an extra regressor: it states explicitly how the
noisy observable x_t relates to the latent h_t, which is what lets both
equations be estimated jointly rather than plugging x in as if it were exact.

Persistence is beta + phi*gamma (substitute the measurement equation into the
GARCH equation), not beta alone.

WHICH REALIZED MEASURE
----------------------
Any PER-DAY variance proxy: pass the series to the constructor. The measure is
part of the specification, not an evaluator-level setting — RealGARCH-GK and
RealGARCH-RS-overnight are different models and belong in the results table as
different rows, exactly as GARCH-NORMAL and GARCH-T do. This is also why `fit`
ignores its `target` argument: the notebook-level ML training target must never
silently redefine what this model is.

Yang-Zhang CANNOT be used. It is a multi-day estimator, so a rolling YZ at date
t is the average variance over the preceding window rather than day t's
variance. The measurement equation would then relate h_t to an average of the
last N days: phi would absorb the smoothing, u_t would be heavily
autocorrelated in violation of its own assumption, and the premise that x_t is
a sharper signal of h_t than r_t^2 would be lost. Use
`rogers_satchell_overnight` for a drift-robust, overnight-inclusive per-day
measure instead.

A NOTE ON COMPARING LIKELIHOODS
-------------------------------
The joint likelihood scores BOTH equations, so it is NOT comparable with the
return-only likelihood of a standard GARCH — it is evaluating a different number
of observed series. Hansen et al. handle this with the PARTIAL likelihood, the
return-equation term alone, which is directly comparable. `info_criteria()`
reports the partial one under the same keys the rest of the pipeline uses, so a
joint table stays honest, and exposes the joint figures under separate keys.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

_PARAM_NAMES = ("omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "sigma_u")
_LOG_2PI = float(np.log(2.0 * np.pi))


def _filter_log_h(theta: np.ndarray, log_x: np.ndarray, log_h0: float) -> np.ndarray:
    """
    Run the GARCH recursion and return log h for t = 0 .. T (length T+1).

    The recursion log h_t = omega + beta*log h_{t-1} + gamma*log x_{t-1} is a
    linear AR(1) whose forcing term depends only on OBSERVED data, not on the
    model's own output, so it is a plain IIR filter rather than something that
    needs stepping in Python. At ~100 re-estimations per model with several
    hundred likelihood evaluations each, that difference is the difference
    between seconds and minutes.

    The final element is log h_{T}, the one-step-ahead forecast state.
    """
    omega, beta, gamma = theta[0], theta[1], theta[2]
    v = omega + gamma * log_x                      # forcing term, length T
    out, _ = lfilter([1.0], [1.0, -beta], v, zi=[beta * log_h0])
    return np.concatenate(([log_h0], out))


def _neg_loglik(
    theta: np.ndarray, r: np.ndarray, log_x: np.ndarray, log_h0: float
) -> float:
    """Negative JOINT log-likelihood (return equation + measurement equation)."""
    log_h_full = _filter_log_h(theta, log_x, log_h0)
    log_h = log_h_full[:-1]
    if not np.isfinite(log_h).all():
        return 1e12

    xi, phi, tau1, tau2, sigma_u = theta[3], theta[4], theta[5], theta[6], theta[7]
    if sigma_u <= 0:
        return 1e12

    h = np.exp(log_h)
    z = r / np.sqrt(h)
    u = log_x - xi - phi * log_h - (tau1 * z + tau2 * (z * z - 1.0))

    ll_r = -0.5 * np.sum(_LOG_2PI + log_h + z * z)
    ll_x = -0.5 * np.sum(_LOG_2PI + 2.0 * np.log(sigma_u) + (u / sigma_u) ** 2)
    total = ll_r + ll_x
    return 1e12 if not np.isfinite(total) else -total


class RealizedGARCH:
    """
    Log-linear Realized GARCH(1,1) with Gaussian return and measurement errors.

    Implements the pipeline's model protocol — fit / update / forecast_variance —
    so it drops into RollingEvaluator alongside the GARCH, XGB and LSTM families.

    Parameters
    ----------
    realized_measure : per-day realized variance series covering the WHOLE
        history the evaluator will walk over, indexed like the returns. It is
        supplied once, at construction, and every method reindexes it onto the
        index of the returns it is handed, so the model can never see a value
        dated after the data it is being asked to condition on. `partial` this
        in when building factories for RollingEvaluator.
    floor_q : lower quantile at which log x is floored. Garman-Klass is not
        bounded below by zero — BZ=F has 80 days with H == L == O == C and 7
        where the close sits outside [L, H] — and log of a non-positive number
        would poison the whole likelihood. The floor is computed ONCE from the
        fit sample and reused by update(), so the recursion is identical over
        the overlapping span.
    n_starts : number of starting values for the optimiser. The joint likelihood
        is not guaranteed unimodal, and this is re-estimated ~100 times during a
        rolling evaluation where a single bad local optimum would show up as an
        unexplained spike.

    scale : returns are multiplied by this and the realized measure by its
        square before estimation, matching GARCHModel's default of 100. This is
        not cosmetic. A log-likelihood is not scale-free — rescaling the data by
        c shifts it by -n*log(c), which is about -14,300 on a 3,110-day sample
        at c=100 — so a RealizedGARCH fitted on raw returns and a GARCHModel
        fitted on percent returns produce log-likelihoods that differ by
        thousands for reasons having nothing to do with fit. Since the entire
        point of reporting the partial likelihood is comparability with the
        GARCH family, the two must be on the same scale. Variance outputs are
        converted back to original units.

    Because both equations are linear in logs, the scale is absorbed by omega
    and xi (with delta = 2*log(scale): omega -> omega + delta*(1-beta-gamma),
    xi -> xi + delta*(1-phi)), so the dynamics, persistence and forecasts in
    original units are unaffected by the choice.

    A note on omega and xi: they are only WEAKLY identified apart from one
    another, because a level shift c in log h can be absorbed by
    omega -> omega + c*(1-beta) and xi -> xi - phi*c. On simulated data the
    individual estimates drift while their combination, and therefore the
    implied unconditional variance, is recovered accurately (implied E[log h]
    within 0.1 of truth, corr(fitted log h, true log h) = 0.9999). Read the
    persistence and the fitted variance, not xi in isolation.
    """

    def __init__(
        self,
        realized_measure: pd.Series,
        *,
        scale: float = 100.0,
        floor_q: float = 0.001,
        n_starts: int = 2,
        maxiter: int = 500,
    ) -> None:
        if not isinstance(realized_measure, pd.Series):
            raise TypeError(
                "realized_measure must be a pd.Series indexed like the returns; "
                f"got {type(realized_measure).__name__}."
            )
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}.")
        self.realized_measure = realized_measure
        self.scale = scale
        self.floor_q = floor_q
        self.n_starts = n_starts
        self.maxiter = maxiter

        self._theta: np.ndarray | None = None
        self._x_floor: float | None = None
        self._log_h0: float | None = None
        self._log_h_next: float | None = None   # state: log h for the next step
        self._nobs: int = 0
        self._ll_joint: float = np.nan
        self._ll_partial: float = np.nan
        self._insample_log_h: pd.Series | None = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _aligned_log_x(self, returns: pd.Series) -> np.ndarray:
        """
        The realized measure for exactly the dates in `returns`, floored and
        logged. Reindexing onto returns.index — rather than slicing the stored
        series positionally — is what makes look-ahead structurally impossible.
        """
        x = self.realized_measure.reindex(returns.index)
        if x.isna().any():
            n = int(x.isna().sum())
            raise ValueError(
                f"realized_measure is missing {n} of {len(x)} dates present in "
                f"the returns being fitted. It must cover the whole evaluation "
                f"history; pass the full proxy series, not a slice."
            )
        scaled = np.asarray(x, dtype=float) * (self.scale ** 2)
        return np.log(np.maximum(scaled, self._x_floor))

    # E[log z^2] for z ~ N(0, 1): digamma(1/2) + log 2. Needed to turn the
    # observable mean(log r^2) into an estimate of mean(log h), since
    # log r^2 = log h + log z^2.
    _E_LOG_CHI2_1 = -1.2703628454614782

    def _starting_values(self, r: np.ndarray, log_x: np.ndarray) -> list[np.ndarray]:
        """
        omega and xi are only weakly identified apart from one another — a level
        shift c in log h can be absorbed by omega -> omega + c(1-beta) and
        xi -> xi - phi*c — so the optimiser benefits from being started near the
        right level rather than at an arbitrary one. mean(log r^2) gives it:
        log r^2 = log h + log z^2, so mean(log h) ~ mean(log r^2) + 1.2704, and
        xi ~ mean(log x) - mean(log h) follows.
        """
        mean_log_x = float(np.mean(log_x))
        r2 = r * r
        positive = r2[r2 > 0]
        mean_log_h = (
            float(np.mean(np.log(positive))) - self._E_LOG_CHI2_1
            if positive.size else mean_log_x
        )
        xi0 = float(np.clip(mean_log_x - mean_log_h, -5.0, 5.0))
        resid_sd = float(np.std(log_x)) or 0.5

        starts = []
        for beta, gamma, phi in ((0.60, 0.35, 1.0), (0.40, 0.55, 1.0)):
            omega = mean_log_h * (1.0 - beta - phi * gamma) - gamma * xi0
            starts.append(np.array(
                [omega, beta, gamma, xi0, phi, -0.05, 0.05, max(resid_sd, 1e-3)],
                dtype=float,
            ))
        return starts[: max(1, self.n_starts)]

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series, target=None) -> "RealizedGARCH":
        """
        Joint maximum likelihood over the return and measurement equations.

        `target` is accepted for interface compatibility with the ML models and
        is IGNORED. The realized measure is a defining part of this
        specification and comes from the constructor, so that changing the
        notebook's ML training target cannot silently turn this into a
        different model.
        """
        r = np.asarray(returns, dtype=float) * self.scale
        if len(r) < 50:
            raise ValueError(f"RealizedGARCH needs at least 50 observations, got {len(r)}.")

        x_raw = self.realized_measure.reindex(returns.index)
        if x_raw.isna().any():
            raise ValueError(
                f"realized_measure is missing {int(x_raw.isna().sum())} of "
                f"{len(x_raw)} dates in the fit window; pass the full series."
            )
        positive = np.asarray(x_raw, dtype=float) * (self.scale ** 2)
        positive = positive[positive > 0]
        if positive.size == 0:
            raise ValueError("realized_measure has no positive values in the fit window.")
        self._x_floor = float(np.quantile(positive, self.floor_q))

        log_x = self._aligned_log_x(returns)
        self._log_h0 = float(np.mean(log_x))

        bounds = [
            (-10.0, 10.0),    # omega
            (0.0, 0.999),     # beta
            (0.0, 2.0),       # gamma
            (-10.0, 10.0),    # xi
            (0.0, 3.0),       # phi
            (-2.0, 2.0),      # tau1
            (-2.0, 2.0),      # tau2
            (1e-4, 10.0),     # sigma_u
        ]
        # beta + phi*gamma < 1 is the stationarity condition for log h; without
        # it the optimiser can wander into an explosive region where the
        # likelihood is still finite on this sample but the forecasts diverge.
        constraints = ({"type": "ineq",
                        "fun": lambda th: 0.999 - (th[1] + th[4] * th[2])},)

        best, best_nll = None, np.inf
        for x0 in self._starting_values(r, log_x):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _neg_loglik, x0, args=(r, log_x, self._log_h0),
                    method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": self.maxiter, "ftol": 1e-9},
                )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best, best_nll = res.x, float(res.fun)

        if best is None:
            raise RuntimeError("RealizedGARCH: joint MLE failed from every starting value.")

        self._theta = best
        self._nobs = len(r)
        self._store_state(r, log_x, returns.index)
        return self

    def update(self, returns: pd.Series) -> "RealizedGARCH":
        """
        Roll both recursions forward at fixed parameters.

        Re-filters from the same origin as the last fit, so within a re-fit
        block the state is produced by exactly the recursion the parameters
        were estimated under.
        """
        self._require_fitted()
        r = np.asarray(returns, dtype=float) * self.scale
        log_x = self._aligned_log_x(returns)
        self._store_state(r, log_x, returns.index)
        return self

    def _store_state(self, r: np.ndarray, log_x: np.ndarray, index) -> None:
        log_h_full = _filter_log_h(self._theta, log_x, self._log_h0)
        log_h = log_h_full[:-1]
        self._log_h_next = float(log_h_full[-1])
        self._insample_log_h = pd.Series(log_h, index=index, name="log_h")

        xi, phi, tau1, tau2, sigma_u = self._theta[3:8]
        h = np.exp(log_h)
        z = r / np.sqrt(h)
        u = log_x - xi - phi * log_h - (tau1 * z + tau2 * (z * z - 1.0))
        self._ll_partial = float(-0.5 * np.sum(_LOG_2PI + log_h + z * z))
        self._ll_joint = self._ll_partial + float(
            -0.5 * np.sum(_LOG_2PI + 2.0 * np.log(sigma_u) + (u / sigma_u) ** 2)
        )

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        """
        Conditional variance forecasts, from the state left by fit()/update().

        One step ahead is exact: log h_{T+1} is already determined by data
        observed through T. Beyond that the realized measure is unknown, so it
        is replaced by its conditional expectation E[log x] = xi + phi*log h,
        giving log h_{T+k} = (omega + gamma*xi) + (beta + phi*gamma)*log h_{T+k-1}.
        """
        self._require_fitted()
        omega, beta, gamma, xi, phi = self._theta[:5]
        out = np.empty(horizon)
        log_h = self._log_h_next
        out[0] = np.exp(log_h)
        for k in range(1, horizon):
            log_h = (omega + gamma * xi) + (beta + phi * gamma) * log_h
            out[k] = np.exp(log_h)
        return out / (self.scale ** 2)

    # ------------------------------------------------------------------
    # In-sample
    # ------------------------------------------------------------------

    def insample_variance(self) -> pd.Series:
        """Filtered conditional variance h_t over the fit sample."""
        self._require_fitted()
        return (np.exp(self._insample_log_h) / (self.scale ** 2)).rename(
            "insample_variance"
        )

    @property
    def params(self) -> pd.Series:
        self._require_fitted()
        return pd.Series(self._theta, index=list(_PARAM_NAMES))

    @property
    def persistence(self) -> float:
        """beta + phi*gamma — the decay rate of log h, not beta alone."""
        self._require_fitted()
        return float(self._theta[1] + self._theta[4] * self._theta[2])

    @property
    def loglikelihood(self) -> float:
        """
        PARTIAL (return-equation) log-likelihood.

        This is the one that is comparable with a standard GARCH, which only
        ever models the returns. The joint figure scores an extra observed
        series and is therefore on a different footing; it is available as
        `joint_loglikelihood`.
        """
        self._require_fitted()
        return self._ll_partial

    @property
    def joint_loglikelihood(self) -> float:
        """Joint log-likelihood of the return AND measurement equations."""
        self._require_fitted()
        return self._ll_joint

    @property
    def n_params(self) -> int:
        return len(_PARAM_NAMES)

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglikelihood

    @property
    def bic(self) -> float:
        return self.n_params * float(np.log(self._nobs)) - 2.0 * self.loglikelihood

    def info_criteria(self) -> dict:
        """
        AIC / BIC / LogL on the PARTIAL likelihood, so these keys mean the same
        thing here as they do for every other model in a joint table, plus the
        joint figures under their own names.

        The partial criteria still charge all eight parameters against a
        return-only fit, which is the conservative direction: it cannot flatter
        this model relative to a plain GARCH.
        """
        self._require_fitted()
        return {
            "AIC": self.aic,
            "BIC": self.bic,
            "LogL": self.loglikelihood,
            "LogL_joint": self.joint_loglikelihood,
            "AIC_joint": 2.0 * self.n_params - 2.0 * self.joint_loglikelihood,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._theta is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        name = self.realized_measure.name or "custom"
        if self._theta is None:
            return f"RealizedGARCH(realized_measure={name!r}, <unfitted>)"
        return (
            f"RealizedGARCH(realized_measure={name!r}, "
            f"persistence={self.persistence:.4f}, LogL={self.loglikelihood:.2f})"
        )


def make_realized_garch(realized_measure: pd.Series, **kwargs) -> RealizedGARCH:
    """
    Module-level factory, picklable for RollingEvaluator's process-parallel
    path (a lambda is not). Use with functools.partial:

        partial(make_realized_garch, gk_series)
    """
    return RealizedGARCH(realized_measure, **kwargs)
