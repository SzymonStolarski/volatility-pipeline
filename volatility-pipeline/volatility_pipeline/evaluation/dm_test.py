from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from dataclasses import dataclass


@dataclass
class DMResult:
    model1: str
    model2: str
    statistic: float   # HLN-corrected MDM statistic
    pvalue: float      # two-sided p-value from t(T-1)
    loss: str
    horizon: int
    n: int             # number of forecast periods

    @property
    def better_model(self) -> str | None:
        """
        The model with significantly lower loss at 5% level.
        Negative statistic → model1 has lower loss on average.
        Returns None if H0 (equal accuracy) is not rejected.
        """
        if self.pvalue >= 0.05:
            return None
        return self.model1 if self.statistic < 0 else self.model2

    def __repr__(self) -> str:
        sig = "*" if self.pvalue < 0.05 else ""
        return (
            f"DMResult({self.model1!r} vs {self.model2!r}): "
            f"MDM={self.statistic:+.4f}, p={self.pvalue:.4f}{sig}"
        )


def diebold_mariano_hln(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0: equal predictive accuracy  (E[d_t] = 0)
    H1: d_t ≠ 0  (two-sided)

    Parameters
    ----------
    e1, e2 : forecast errors (forecast − actual) for model 1 and model 2
    h      : forecast horizon (1 for one-step-ahead)
    loss   : 'squared' or 'absolute'
             For QLIKE use diebold_mariano_from_losses() with pre-computed losses.

    Returns
    -------
    (mdm_statistic, pvalue)
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(
            f"loss must be 'squared' or 'absolute' when passing errors. "
            f"For QLIKE use diebold_mariano_from_losses(). Got: {loss!r}"
        )

    return _hln_from_d(d, h)


def diebold_mariano_from_losses(
    l1: np.ndarray,
    l2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """
    DM test from pre-computed per-period loss series.

    d_t = l1_t − l2_t
    Positive d_bar → model 2 is on average more accurate.

    Returns
    -------
    (mdm_statistic, pvalue)
    """
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    return _hln_from_d(d, h)


def _hln_from_d(d: np.ndarray, h: int) -> tuple[float, float]:
    """Core HLN computation given the loss differential series d."""
    T = len(d)
    d_bar = d.mean()

    # Long-run variance via sample autocovariances at lags 0 … h-1
    d_c = d - d_bar
    gamma = np.array([np.mean(d_c * np.roll(d_c, k)) for k in range(h)])
    V_d = (gamma[0] + 2.0 * gamma[1:].sum()) / T

    if V_d <= 0:
        V_d = float(np.var(d, ddof=0)) / T  # fallback

    dm_stat = d_bar / np.sqrt(V_d)

    # HLN small-sample correction factor  √[(T+1−2h+h(h-1)/T) / T]
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    mdm_stat = float(correction * dm_stat)

    pvalue = float(2.0 * t_dist.sf(abs(mdm_stat), df=T - 1))
    return mdm_stat, pvalue


def dm_matrix(
    results: dict,
    h: int = 1,
    loss: str = "squared",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all pairwise DM (HLN-corrected) statistics and p-values.

    Parameters
    ----------
    results : dict mapping model name → ForecastResult
    h       : forecast horizon
    loss    : loss function for comparing models

    Returns
    -------
    (stat_df, pval_df) — upper-triangular DataFrames of MDM statistics and p-values
    """
    names = list(results.keys())
    n = len(names)
    stat_df = pd.DataFrame(np.nan, index=names, columns=names)
    pval_df = pd.DataFrame(np.nan, index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            l1 = results[names[i]].loss_series(loss).values
            l2 = results[names[j]].loss_series(loss).values
            stat, pval = diebold_mariano_from_losses(l1, l2, h=h)
            stat_df.loc[names[i], names[j]] = stat
            pval_df.loc[names[i], names[j]] = pval

    return stat_df, pval_df
