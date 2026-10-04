"""
Contract-roll days in continuous front-month futures series.

WHY THIS EXISTS
---------------
Yahoo's NG=F and BZ=F are unadjusted chains of front-month contracts. On the
first trading day after the front contract expires, the series switches to the
next contract: the prior close belongs to the old contract, while that day's
open, high, low and close belong to the new one. Every quantity that compares a
price from day t with a price from day t-1 therefore contains the calendar
spread between two different contracts:

  * the overnight gap ln(O_t / C_{t-1}), and so every *_overnight proxy;
  * the close-to-close return ln(C_t / C_{t-1}), and so GARCH estimation and
    its variance recursion, the ML lag features, and the r^2 proxy.

The worst case in this project's sample is 2026-01-29 on NG=F: the February
contract settled at 7.460 and the March contract opened at 3.742, a recorded
"return" of about -64% on a day nothing of the sort happened. Intraday-only
quantities (Garman-Klass, ln(C_t / O_t)) are immune, because every price they
use comes from the same contract.

Measured on 2010-01 -> 2026-06: roll days are 4.8% of trading days but carry
67% of all overnight variance on NG=F and 23% on BZ=F, and 29% of the
test-window sum of r^2 on NG=F.

HOW ROLL DAYS ARE IDENTIFIED: FROM THE EXCHANGE CALENDAR, NOT FROM THE DATA
---------------------------------------------------------------------------
A threshold on the size of the gap (the old 15% screen) is a guess about the
data: it misses rolls with a small spread and mislabels genuine overnight moves
(NG=F 2026-02-02, a -16% Monday gap, is not a roll). The expiry rules are fixed
by the exchange and known in advance, so they identify roll days a priori:

  nymex_ng  Henry Hub natural gas (NYMEX NG). Trading terminates on the third
            business day before the first calendar day of the delivery month;
            the roll day is the next trading day.
  nymex_bz  Brent crude, last-day financial (NYMEX BZ). Trading terminates on
            the last business day of the second month before the delivery
            month, so the front month changes on the first trading day of every
            calendar month.

Business days are taken from the data's own trading calendar (the dates
present in the OHLC index), which already omits exchange holidays.

Validation on the project's data: 7 of the 8 NG=F days with |overnight gap|
> 15% fall exactly on nymex_ng roll days, and the eighth (2026-02-02) is a
genuine Monday gap. BZ=F has no gap above 15%, but its mean |gap| on nymex_bz
roll days is 2.4x the mean on other days. `roll_day_report` reprints this
check on every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_datetime_index(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("calendar_roll_days needs a sorted index of unique trading dates.")
    return idx


def _nymex_ng(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading day after the 3rd business day before each delivery month."""
    rolls = []
    first_month = idx[0].to_period("M") + 1
    last_month = idx[-1].to_period("M") + 2
    for month in pd.period_range(first_month, last_month, freq="M"):
        before = idx[idx < month.to_timestamp()]
        if len(before) < 3:
            continue
        expiry = before[-3]
        after = idx[idx > expiry]
        if len(after):
            rolls.append(after[0])
    return pd.DatetimeIndex(sorted(set(rolls)))


def _nymex_bz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of every calendar month."""
    s = pd.Series(idx, index=idx)
    firsts = s.groupby(idx.to_period("M")).min()
    return pd.DatetimeIndex(firsts.values)


_RULES = {"nymex_ng": _nymex_ng, "nymex_bz": _nymex_bz}

#: Yahoo ticker -> exchange expiry rule.
ROLL_RULES: dict[str, str] = {"NG=F": "nymex_ng", "BZ=F": "nymex_bz"}


def resolve_roll_rule(rule: str) -> str:
    """Accept either a ticker from ROLL_RULES or a rule name; return the rule name."""
    if rule in ROLL_RULES:
        return ROLL_RULES[rule]
    if rule in _RULES:
        return rule
    raise ValueError(
        f"no contract-roll rule for {rule!r}. Known tickers: {sorted(ROLL_RULES)}; "
        f"known rules: {sorted(_RULES)}. Add the instrument's exchange expiry rule "
        f"here, or run with roll_handling='none' if the series has no rolls."
    )


def calendar_roll_days(index, rule: str) -> pd.DatetimeIndex:
    """
    Calendar roll days that fall inside `index`.

    `index` is the trading calendar of the OHLC data (DatetimeIndex, or a
    PeriodIndex, which is converted). `rule` is a ticker in ROLL_RULES or a rule
    name ('nymex_ng', 'nymex_bz').

    The first date of `index` is never returned: a roll is defined relative to
    the previous close, and the first date has none inside the sample.
    """
    idx = _as_datetime_index(index)
    if len(idx) < 2:
        return pd.DatetimeIndex([])
    days = _RULES[resolve_roll_rule(rule)](idx)
    days = days[days.isin(idx) & (days != idx[0])]
    return days.rename("roll_day")


def roll_adjusted_returns(
    open_: pd.Series,
    close: pd.Series,
    roll_days: pd.DatetimeIndex,
) -> pd.Series:
    """
    Log returns with the contract spread removed on roll days.

    On a roll day the close-to-close return ln(C_t / C_{t-1}) compares two
    different contracts. It is replaced by the open-to-close return
    ln(C_t / O_t), in which both prices belong to the new contract. Every other
    day keeps its close-to-close return. The day after a roll needs nothing: its
    prior close is the roll day's close, already the new contract.

    What this gives up is the genuine overnight move on roll days, which cannot
    be separated from the spread without the new contract's prior-day price
    (a full ratio back-adjustment needs quotes of both contracts, which Yahoo
    does not provide). On NG=F that is about 0.6% of total variance.

    Pass the FULL OHLC series; the result starts on the second date, like
    np.log(close / close.shift(1)).dropna().
    """
    r = np.log(close / close.shift(1)).dropna()
    on_roll = r.index.isin(pd.DatetimeIndex(roll_days))
    intraday = np.log(close / open_).reindex(r.index)
    return r.where(~on_roll, intraday).rename("returns")


def roll_day_report(
    open_: pd.Series,
    close: pd.Series,
    roll_days: pd.DatetimeIndex,
    threshold: float = 0.15,
    split=None,
) -> dict:
    """
    Check the calendar against the data, and size the problem.

    Returns a dict with
      n_roll_days                  roll days inside the sample (and n_train /
                                   n_after_split when `split` is given)
      mean_abs_gap_roll / _other   mean |ln(O_t / C_{t-1})| on and off roll days
      share_overnight_var_on_roll  fraction of the summed squared overnight gaps
                                   that falls on roll days
      large_gaps                   DataFrame of days with |gap| > threshold and
                                   whether each is a calendar roll day
      n_large_on_roll / n_large_off_roll

    A calendar that is right puts almost every large gap on a roll day; a large
    gap off the calendar is either genuine news or a sign the rule is wrong for
    this series, and should be looked at.
    """
    rd = pd.DatetimeIndex(roll_days)
    gap = np.log(open_ / close.shift(1)).dropna()
    on_roll = gap.index.isin(rd)
    large = gap.abs() > threshold
    large_gaps = pd.DataFrame({
        "overnight_gap": gap[large],
        "on_roll_day": on_roll[large.to_numpy()],
    })
    out = {
        "threshold": float(threshold),
        "n_roll_days": int(on_roll.sum()),
        "mean_abs_gap_roll": float(gap[on_roll].abs().mean()) if on_roll.any() else float("nan"),
        "mean_abs_gap_other": float(gap[~on_roll].abs().mean()),
        "share_overnight_var_on_roll": float((gap[on_roll] ** 2).sum() / (gap ** 2).sum()),
        "large_gaps": large_gaps,
        "n_large_on_roll": int(large_gaps["on_roll_day"].sum()),
        "n_large_off_roll": int((~large_gaps["on_roll_day"]).sum()),
    }
    if split is not None:
        split = pd.Timestamp(split)
        out["n_train"] = int(((gap.index <= split) & on_roll).sum())
        out["n_after_split"] = int(((gap.index > split) & on_roll).sum())
    return out
