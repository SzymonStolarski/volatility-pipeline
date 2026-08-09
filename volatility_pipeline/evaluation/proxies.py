from __future__ import annotations
import numpy as np
import pandas as pd


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Garman-Klass (1980) daily variance estimator.

    σ² = 0.5·(ln H/L)² − (2·ln2 − 1)·(ln C/O)²

    Uses the full intraday price range, which is 5-8× less noisy than squared
    returns. Assumes no overnight gap.

    NOT guaranteed non-negative: the -(2·ln2-1)·(ln C/O)² term dominates
    whenever the close-to-open move is large relative to the recorded range,
    which happens when the OHLC record is internally inconsistent. BZ=F has 80
    days with H == L == O == C (proxy exactly 0, alongside a nonzero
    close-to-close return) and 7 where the close sits outside [L, H] (proxy
    negative), all before 2020. Callers that use this as a regression target
    must floor it; see models/targets.log_variance_target.
    """
    ln_hl = np.log(high.values / low.values)
    ln_co = np.log(close.values / open_.values)
    gk = 0.5 * ln_hl ** 2 - (2.0 * np.log(2) - 1.0) * ln_co ** 2
    return pd.Series(gk, index=close.index, name="garman_klass")


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson (1980) daily variance estimator.

    σ² = (ln H/L)² / (4·ln2)

    Uses only the High-Low range; simpler than GK but ignores the
    open-to-close component and overnight gaps.
    """
    ln_hl = np.log(high.values / low.values)
    pk = ln_hl ** 2 / (4.0 * np.log(2))
    return pd.Series(pk, index=high.index, name="parkinson")


def squared_returns(returns: pd.Series) -> pd.Series:
    """Squared log-returns — the simplest but noisiest variance proxy."""
    return (returns ** 2).rename("squared_returns")
