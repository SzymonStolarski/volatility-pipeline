"""
Realized-variance proxies built from daily OHLC data.

WHAT EACH PROXY MEASURES, AND WHY IT MATTERS HERE
-------------------------------------------------
The GARCH family is estimated by QMLE on close-to-close returns, so it
forecasts the FULL daily variance: the overnight gap plus the intraday session.
Garman-Klass and Parkinson measure the INTRADAY session only — they never cross
the overnight gap. Scoring a close-to-close forecast against an intraday-only
proxy therefore penalises the model for predicting a larger quantity than the
one it is compared with, by whatever share of total variance happens overnight.

Measured on this project's data (test window 2022-07 -> 2026-06):
    Brent BZ=F   mean GK / mean r^2 = 1.0366
    Henry Hub NG=F                  = 0.6295
so the mismatch is negligible on crude in this period and large on gas.

The proxies below therefore come in two families:

  INTRADAY ONLY   garman_klass, parkinson, rogers_satchell
  FULL VARIANCE   garman_klass_overnight, rogers_satchell_overnight,
                  squared_returns, yang_zhang

Use a FULL-VARIANCE proxy when scoring GARCH forecasts, unless there is a
specific reason to isolate the intraday component.

A WARNING SPECIFIC TO FUTURES
-----------------------------
Every full-variance proxy except `squared_returns` reads the overnight gap
ln(O_t / C_{t-1}), and on a front-month futures series that gap is contaminated
by contract rolls: the "overnight return" on a roll date is the price
difference between two different contracts, not a price move. On NG=F, eight
days with |gap| > 15% carry 47.4% of the entire overnight sum; the worst
(2026-01-29, prior close 7.460 -> open 3.742) implies an overnight variance of
0.476 against a Garman-Klass reading of 5.76e-4 for the same day, a factor of
about 830. Garman-Klass is immune because it never crosses the gap.

The roll days are identified from the exchange calendar in
`volatility_pipeline.data.rolls`, and `compute_proxy(..., roll_days=...)` drops
the overnight term on those days. The same spread also sits in the
close-to-close RETURNS, so the returns need the matching treatment; build every
series through `volatility_pipeline.data.prepare_series`, which applies one rule
to the returns, every proxy and the floor at once.

ON YANG-ZHANG
-------------
Yang & Zhang (2000) is a MULTI-DAY estimator: two of its three components are
sample variances computed ACROSS days, so it returns the average daily variance
over a window, not a variance for one day. It is provided here for descriptive
comparison and is NOT a drop-in per-day proxy — see the note in `yang_zhang`.
For a per-day full-variance proxy use `rogers_satchell_overnight` (drift-robust)
or `garman_klass_overnight`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Intraday-only estimators (one value per day; the overnight gap is not seen)
# ---------------------------------------------------------------------------

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
    returns. Assumes no overnight gap and zero drift.

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


def rogers_satchell(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Rogers-Satchell (1991) daily variance estimator.

    σ² = u(u − c) + d(d − c),  where u = ln H/O, d = ln L/O, c = ln C/O

    Like Garman-Klass this measures the intraday session only, but unlike
    Garman-Klass and Parkinson it is DRIFT-INDEPENDENT: it stays unbiased when
    the price has a non-zero mean return over the session, whereas GK and
    Parkinson are biased upward by drift because they attribute the drift-driven
    part of the range to volatility. Over a long sample with a trend — energy
    futures in 2020-2022, say — that difference is not negligible.

    Non-negative for internally consistent OHLC (since L <= O, C <= H implies
    u >= max(0, c) and d <= min(0, c)), but the same bad records that make
    Garman-Klass negative can do so here; floor it before using it as a
    regression target.
    """
    u = np.log(high.values / open_.values)
    d = np.log(low.values / open_.values)
    c = np.log(close.values / open_.values)
    rs = u * (u - c) + d * (d - c)
    return pd.Series(rs, index=close.index, name="rogers_satchell")


# ---------------------------------------------------------------------------
# The overnight component
# ---------------------------------------------------------------------------

def overnight_variance(
    open_: pd.Series,
    close: pd.Series,
    roll_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Squared overnight (close-to-open) return, (ln O_t / C_{t-1})².

    This is the piece of the daily close-to-close variance that every
    range-based intraday estimator omits. The first observation is NaN, since
    it has no preceding close.

    Pass the FULL close series, not one already reindexed onto the returns
    index: the returns series drops its own first day, so reindexing before
    shifting would silently discard the first usable overnight observation.

    On futures, a roll date's gap is a change of contract, not a price move.
    Pass `roll_days` to set the term to 0 on those dates: what remains for the
    day is the intraday session, which lies entirely within the new contract.
    """
    prev_close = close.shift(1)
    on = np.log(open_.values / prev_close.values) ** 2
    out = pd.Series(on, index=close.index, name="overnight_variance")
    if roll_days is not None:
        on_roll = close.index.isin(pd.DatetimeIndex(roll_days)) & out.notna().to_numpy()
        out = out.mask(on_roll, 0.0)
    return out


# ---------------------------------------------------------------------------
# Full-variance estimators (one value per day)
# ---------------------------------------------------------------------------
#
# These simply ADD the squared overnight return to an intraday estimator. That
# is the right per-day construction: the daily close-to-close variance is the
# sum of the overnight and intraday components, and each part is estimated
# without bias, so the sum is too.
#
# Note that Yang-Zhang's k-weighting is deliberately NOT applied here. Those
# weights exist to minimise the VARIANCE OF THE ESTIMATOR across a multi-day
# window; there is no such optimisation to perform for a single day, and
# applying the weights to one day's data would make the result biased for that
# day's variance rather than more efficient.

def garman_klass_overnight(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    roll_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Garman-Klass plus the squared overnight return — a per-day estimator of the
    FULL close-to-close variance, and so directly comparable with what a GARCH
    model estimated on close-to-close returns forecasts.

    First observation is NaN (no preceding close for the overnight term).
    `roll_days`: see `overnight_variance`.
    """
    gk = garman_klass(open_, high, low, close)
    on = overnight_variance(open_, close, roll_days)
    return (gk + on).rename("garman_klass_overnight")


def rogers_satchell_overnight(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    roll_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Rogers-Satchell plus the squared overnight return.

    The per-day full-variance proxy this module recommends by default: it covers
    the overnight gap that GK omits AND stays unbiased under drift, which GK
    does not. It is also the closest per-day analogue of what Yang-Zhang
    estimates over a window, without Yang-Zhang's multi-day construction.

    First observation is NaN (no preceding close for the overnight term).
    `roll_days`: see `overnight_variance`.
    """
    rs = rogers_satchell(open_, high, low, close)
    on = overnight_variance(open_, close, roll_days)
    return (rs + on).rename("rogers_satchell_overnight")


def squared_returns(returns: pd.Series) -> pd.Series:
    """Squared log-returns — the simplest but noisiest variance proxy."""
    return (returns ** 2).rename("squared_returns")


# ---------------------------------------------------------------------------
# Yang-Zhang (multi-day)
# ---------------------------------------------------------------------------

def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Yang & Zhang (2000) drift-independent variance estimator, computed on a
    rolling window of `window` days.

    σ²_YZ = V_o + k·V_c + (1 − k)·V_RS
        V_o  = sample variance of o_i = ln(O_i / C_{i-1})   across the window
        V_c  = sample variance of c_i = ln(C_i / O_i)       across the window
        V_RS = mean of the per-day Rogers-Satchell estimate across the window
        k    = 0.34 / (1.34 + (n+1)/(n-1))

    NOT A PER-DAY PROXY — DO NOT PASS THIS TO RollingEvaluator
    ----------------------------------------------------------
    V_o and V_c are sample variances computed ACROSS the days in the window, so
    what comes back at date t is the AVERAGE daily variance over the preceding
    `window` days, not the realized variance OF day t. Two consequences make it
    unsuitable for scoring one-step-ahead forecasts:

      1. Consecutive values share window-1 of their inputs, so the series is
         heavily autocorrelated and far smoother than the quantity being
         forecast. A smoothed target systematically flatters smooth forecasts,
         which is the same class of error this module's docstring warns about,
         only pointing the other way.
      2. It is a backward-looking average, so it is not the realization of the
         variance the model predicted for that day.

    Provided for descriptive comparison — reporting the level of volatility on
    the estimator the literature regards as most efficient, and quantifying how
    far the per-day proxies sit from it. For scoring, use
    `rogers_satchell_overnight` or `garman_klass_overnight`, which estimate the
    same total variance one day at a time.

    The `k` weights minimise the estimator's variance under the model's
    assumptions; violating the independence assumption behind them makes the
    estimator slightly inefficient, not invalid.

    The first `window` values are NaN.
    """
    if window < 2:
        raise ValueError(
            f"yang_zhang needs window >= 2 to form the cross-day sample "
            f"variances V_o and V_c; got {window}. For a single-day estimate "
            f"use rogers_satchell_overnight."
        )

    o = pd.Series(np.log(open_.values / close.shift(1).values), index=close.index)
    c = pd.Series(np.log(close.values / open_.values), index=close.index)
    rs = rogers_satchell(open_, high, low, close)

    n = float(window)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    v_o = o.rolling(window).var(ddof=1)
    v_c = c.rolling(window).var(ddof=1)
    v_rs = rs.rolling(window).mean()

    yz = v_o + k * v_c + (1.0 - k) * v_rs
    return yz.rename(f"yang_zhang_{window}d")


# ---------------------------------------------------------------------------
# Data-quality diagnostic for the overnight component
# ---------------------------------------------------------------------------

def overnight_gap_report(
    open_: pd.Series,
    close: pd.Series,
    threshold: float = 0.15,
) -> dict:
    """
    Quantify how much of the overnight variance comes from a handful of extreme
    gaps — on a front-month futures series, overwhelmingly contract rolls.

    Any proxy that includes the overnight term inherits these days in full, and
    a single roll can outweigh years of genuine overnight moves: the squared
    term makes a 69% "return" contribute about 0.48 to a series whose typical
    daily variance is around 1e-3.

    Returns a dict with `n_flagged`, `share_of_overnight_sum` (the fraction of
    Σ overnight variance carried by the flagged days), `mean_with` /
    `mean_without` (mean overnight variance including and excluding them), and
    `flagged`, a DataFrame of the offending dates with the prior close, the
    open, and the implied overnight return, worst first.

    A large `share_of_overnight_sum` is a reason to roll-adjust the prices or to
    exclude those dates from the loss, NOT a reason to go back to an
    intraday-only proxy: the overnight variance is real, the contract change is
    not.
    """
    on = overnight_variance(open_, close)
    gap = pd.Series(np.log(open_.values / close.shift(1).values), index=close.index)

    valid = on.notna()
    flagged_mask = valid & (gap.abs() > threshold)

    total = float(on[valid].sum())
    flagged_sum = float(on[flagged_mask].sum())

    flagged = pd.DataFrame({
        "prev_close": close.shift(1)[flagged_mask],
        "open": open_[flagged_mask],
        "overnight_return": gap[flagged_mask],
        "overnight_variance": on[flagged_mask],
    }).sort_values("overnight_variance", ascending=False)

    kept = valid & ~flagged_mask
    return {
        "threshold": float(threshold),
        "n_total": int(valid.sum()),
        "n_flagged": int(flagged_mask.sum()),
        "share_of_overnight_sum": (flagged_sum / total) if total > 0 else float("nan"),
        "mean_with": float(on[valid].mean()),
        "mean_without": float(on[kept].mean()) if kept.any() else float("nan"),
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: name -> (callable, needs_ohlc, includes_overnight)
PROXY_REGISTRY: dict[str, tuple] = {
    "garman_klass":             (garman_klass,             True,  False),
    "parkinson":                (parkinson,                True,  False),
    "rogers_satchell":          (rogers_satchell,          True,  False),
    "garman_klass_overnight":   (garman_klass_overnight,   True,  True),
    "rogers_satchell_overnight": (rogers_satchell_overnight, True, True),
    "squared":                  (squared_returns,          False, True),
}


def compute_proxy(
    name: str,
    *,
    returns: pd.Series,
    open_: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    close: pd.Series | None = None,
    roll_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Build the named per-day variance proxy and align it to `returns.index`.

    Pass the FULL OHLC series as downloaded, not versions already reindexed onto
    the returns index. The overnight-inclusive proxies need the close from the
    day BEFORE the first return, and the returns series has already dropped that
    day; reindexing the inputs first would silently lose one observation and
    leave a NaN the evaluator would then reject.

    `yang_zhang` is deliberately absent from the registry — it is a multi-day
    estimator and must not be used to score one-step-ahead forecasts. Call it
    directly if you want it for description.

    `roll_days` sets the overnight term to 0 on contract-roll dates for the
    *_overnight proxies. Intraday proxies never read the overnight gap and
    ignore it. `squared` is built from the `returns` passed in, so for it the
    returns themselves must already be roll-adjusted — which is why
    `volatility_pipeline.data.prepare_series`, not this function, is the place
    to build a consistent set.
    """
    if name not in PROXY_REGISTRY:
        extra = (
            " Yang-Zhang is a multi-day estimator and is not available as a "
            "scoring proxy; see proxies.yang_zhang."
            if "yang" in name.lower() else ""
        )
        raise ValueError(
            f"unknown variance proxy {name!r}; available: "
            f"{sorted(PROXY_REGISTRY)}.{extra}"
        )

    func, needs_ohlc, includes_overnight = PROXY_REGISTRY[name]

    if not needs_ohlc:
        proxy = func(returns)
    else:
        missing = [n for n, s in
                   (("open", open_), ("high", high), ("low", low), ("close", close))
                   if s is None]
        if missing:
            raise ValueError(
                f"proxy {name!r} is range-based and needs OHLC data; "
                f"missing: {', '.join(missing)}."
            )
        if func is parkinson:
            proxy = func(high, low)
        elif includes_overnight:
            proxy = func(open_, high, low, close, roll_days)
        else:
            proxy = func(open_, high, low, close)

    aligned = proxy.reindex(returns.index)
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        n_bad = int((~np.isfinite(aligned.to_numpy(dtype=float))).sum())
        raise ValueError(
            f"proxy {name!r} has {n_bad} non-finite value(s) after aligning to "
            f"the returns index. If this is 1, the OHLC series probably starts "
            f"on the same day as the returns, leaving the first overnight gap "
            f"undefined — pass the full OHLC history instead of a slice."
        )
    return aligned.rename(name)


# ---------------------------------------------------------------------------
# Floor
# ---------------------------------------------------------------------------

def floor_proxy(proxy: pd.Series, q: float, fit_end) -> tuple[pd.Series, float]:
    """
    Raise every value below a floor to the floor, with the floor fitted on the
    training window only.

    floor = q-quantile of the strictly POSITIVE values dated <= `fit_end`.

    Why positive values only: on BZ=F 1.06% of the training-window
    Garman-Klass+overnight values are <= 0 (80 zero-range OHLC records and a few
    internally inconsistent ones, all before 2020), so a plain 1% quantile is
    0.0 and the floor would floor nothing. Why the training window: the floor is
    a modelling choice and must not use test-period data.

    Applied once, at data preparation, the same floored series serves as the
    scoring proxy, the ML training target and the Realized GARCH measure, so the
    three cannot disagree about any day. On this project's data it changes no
    test-window value on either commodity; it matters for the log transforms in
    the ML target and the Realized GARCH measurement equation.

    Returns (floored series, floor value).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"floor quantile must lie in (0, 1), got {q!r}.")
    fit = proxy[proxy.index <= fit_end]
    positive = fit[fit > 0].to_numpy(dtype=float)
    if positive.size == 0:
        raise ValueError(
            f"proxy {proxy.name!r} has no positive values on or before {fit_end}; "
            f"cannot fit a floor."
        )
    floor = float(np.quantile(positive, q))
    return proxy.clip(lower=floor), floor
