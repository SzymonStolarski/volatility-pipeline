"""
One place to build every series a model sees: returns, scoring proxy, ML
training target, Realized GARCH measures and the r^2 robustness proxy.

WHY ONE PLACE
-------------
The contract-roll spread sits in the overnight gap AND in the close-to-close
return. Cleaning only the proxy leaves the models learning from a jump the
evaluation never sees. On NG=F that does two kinds of damage:

  * Level. GARCH learns the mean of r^2 including the roll jumps, while a
    roll-cleaned proxy excludes them: over the test window the proxy is then
    0.73 x mean r^2, so GARCH looks about 35% too high every day. That is the
    same mechanism that produced the dead Garman-Klass "hybrids win on gas"
    result.
  * Dynamics. After a -64% "return" the GARCH recursion multiplies its forecast
    roughly tenfold and takes weeks to decay, scored against a proxy that says
    nothing happened. GARCH(1,1)-N lost 0.047 QLIKE to this on NG=F.

So the treatment has to be symmetric (agreed with the supervisor, Oct 2026):
the same rule for returns, every proxy and the realized measures. Building
them through one function is how that is guaranteed.

ROLL HANDLING MODES
-------------------
  "adjust"  Main specification. On calendar roll days the return is the
            open-to-close return ln(C_t / O_t) and the overnight term of every
            *_overnight proxy is 0, so both describe the intraday session of
            the new contract. `squared` is built from the adjusted returns.
  "drop"    Robustness check. Roll days are removed from the data entirely —
            from estimation, ML features, every proxy and the out-of-sample
            loss. Returns are computed on the FULL series first and the roll
            row is dropped afterwards: dropping the price instead would turn
            the next day's return into ln(C_{t+1} / C_{t-1}) and re-introduce
            the contract switch one day later. The GARCH recursion and the ML
            lags then treat the removed day as if it did not exist — the same
            approximation GARCH already makes over weekends and holidays.
  "none"    No treatment; reproduces the pre-October (13.09) runs.

THE FLOOR
---------
Each proxy is floored once, at the q-quantile of its positive training-window
values (see `evaluation.proxies.floor_proxy`). Because the ML target and the
realized measure are taken from the same floored series, there is exactly one
floor in the pipeline; the models' own floors should then be set to defer to it
(target_floor_q=0.0 for XGB/LSTM, floor_q=0.0 for RealizedGARCH), which makes
them no-ops on an already floored series.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..evaluation.proxies import compute_proxy, floor_proxy
from .rolls import calendar_roll_days, resolve_roll_rule, roll_adjusted_returns, roll_day_report

ROLL_HANDLING_MODES = ("adjust", "drop", "none")

#: Always built: the r^2 proxy is the robustness yardstick (unbiased for
#: close-to-close variance by construction, if noisy).
ALWAYS_BUILT = ("squared",)


@dataclass
class PreparedSeries:
    """Everything `prepare_series` built, plus what it did to build it."""
    returns: pd.Series
    proxies: dict[str, pd.Series]
    floors: dict[str, float | None]
    roll_days: pd.DatetimeIndex          # calendar roll days inside the sample
    roll_handling: str
    roll_rule: str | None
    train_end: pd.Timestamp
    diagnostics: pd.DataFrame            # per-proxy floor / ratio table
    roll_report: dict | None = field(default=None, repr=False)

    def to_period_index(self, freq: str = "D") -> "PreparedSeries":
        """Copy with returns and proxies on a PeriodIndex, as the arch-based notebooks use."""
        def conv(s: pd.Series) -> pd.Series:
            out = s.copy()
            out.index = pd.PeriodIndex(out.index, freq=freq)
            return out
        return replace(
            self,
            returns=conv(self.returns),
            proxies={k: conv(v) for k, v in self.proxies.items()},
        )

    def describe(self) -> str:
        """Plain-text summary for the notebook's data cell."""
        n_tr = int(self.diagnostics.attrs["roll_days_train"])
        n_te = int(self.diagnostics.attrs["roll_days_after"])
        lines = [
            f"Roll handling: {self.roll_handling}"
            + (f"  (calendar rule {self.roll_rule})" if self.roll_rule else ""),
            f"  calendar roll days in sample: {len(self.roll_days)}"
            f"  (training window {n_tr}, after it {n_te})",
        ]
        if self.roll_handling == "drop":
            lines.append(f"  -> {len(self.roll_days)} roll-day observations removed from every series")
        elif self.roll_handling == "adjust":
            lines.append("  -> on roll days: return = ln(C/O), overnight term of the proxies = 0")
        rr = self.roll_report
        if rr is not None:
            lines.append(
                f"  calendar check: {rr['n_large_on_roll']} of "
                f"{rr['n_large_on_roll'] + rr['n_large_off_roll']} overnight gaps "
                f"> {rr['threshold']:.0%} fall on roll days; mean |gap| on roll days "
                f"{rr['mean_abs_gap_roll']:.4f} vs {rr['mean_abs_gap_other']:.4f} otherwise; "
                f"roll days carry {rr['share_overnight_var_on_roll']:.1%} of overnight variance"
            )
            off = rr["large_gaps"][~rr["large_gaps"]["on_roll_day"]]
            if len(off):
                lines.append("  large gaps NOT on a roll day (genuine moves, kept): "
                             + ", ".join(f"{d.date()} ({g:+.1%})"
                                         for d, g in off["overnight_gap"].items()))
        lines.append(f"Returns: {len(self.returns)} observations, "
                     f"{str(self.returns.index[0])[:10]} -> {str(self.returns.index[-1])[:10]}")
        return "\n".join(lines)


def prepare_series(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    proxies,
    train_end,
    roll_rule: str | None = None,
    roll_handling: str = "adjust",
    floor_q: float | None = 0.01,
) -> PreparedSeries:
    """
    Build returns and every requested proxy under one roll rule and one floor.

    Parameters
    ----------
    open_, high, low, close : the FULL OHLC series as downloaded (DatetimeIndex).
    proxies       : proxy names from PROXY_REGISTRY to build — the scoring
                    proxy, the ML target, the realized measures. Duplicates are
                    fine; 'squared' is always added.
    train_end     : last date of the initial training window. The floor is
                    fitted on data up to here and nowhere else.
    roll_rule     : ticker in ROLL_RULES ('NG=F', 'BZ=F') or rule name. Required
                    unless roll_handling='none', where an unknown instrument only
                    means no roll-day report.
    roll_handling : 'adjust' (main), 'drop' (robustness) or 'none' (pre-October).
    floor_q       : quantile of positive training values used as the floor;
                    None disables the floor.
    """
    if roll_handling not in ROLL_HANDLING_MODES:
        raise ValueError(f"roll_handling must be one of {ROLL_HANDLING_MODES}, got {roll_handling!r}.")
    train_end = pd.Timestamp(train_end)

    rule = None
    if roll_rule is not None:
        try:
            rule = resolve_roll_rule(roll_rule)
        except ValueError:
            # 'none' applies no treatment, so an instrument without a rule only
            # loses the roll-day report; any other mode needs the calendar.
            if roll_handling != "none":
                raise
    elif roll_handling != "none":
        raise ValueError(
            f"roll_handling={roll_handling!r} needs roll_rule (a ticker such as 'NG=F' "
            f"or a rule name) to know which exchange calendar to use."
        )

    rdays = calendar_roll_days(close.index, rule) if rule else pd.DatetimeIndex([])
    raw_returns = np.log(close / close.shift(1)).dropna().rename("returns")
    rdays = rdays[rdays.isin(raw_returns.index)]

    if roll_handling == "adjust":
        returns = roll_adjusted_returns(open_, close, rdays)
    elif roll_handling == "drop":
        returns = raw_returns.drop(rdays)
    else:
        returns = raw_returns

    # In 'drop' mode the roll rows are already gone from `returns`, and
    # compute_proxy aligns to returns.index, so every proxy loses them too. The
    # overnight term on the day AFTER a roll uses the roll day's close, which is
    # the new contract, so nothing else needs touching.
    names = list(dict.fromkeys([*proxies, *ALWAYS_BUILT]))
    built, floors, rows = {}, {}, []
    is_train = returns.index <= train_end
    r2 = returns ** 2
    for name in names:
        p = compute_proxy(
            name, returns=returns, open_=open_, high=high, low=low, close=close,
            roll_days=rdays if roll_handling == "adjust" else None,
        )
        raw = p
        if floor_q is not None:
            p, floors[name] = floor_proxy(p, floor_q, train_end)
        else:
            floors[name] = None
        floored = raw < p
        built[name] = p.rename(name)
        rows.append({
            "proxy": name,
            "floor": floors[name],
            "floored_train": int((floored & is_train).sum()),
            "floored_after": int((floored & ~is_train).sum()),
            "mean/mean r2 (train)": float(p[is_train].mean() / r2[is_train].mean()),
            "mean/mean r2 (after)": float(p[~is_train].mean() / r2[~is_train].mean())
            if (~is_train).any() else np.nan,
        })

    diagnostics = pd.DataFrame(rows).set_index("proxy")
    in_train = rdays <= train_end
    diagnostics.attrs["roll_days_train"] = int(in_train.sum())
    diagnostics.attrs["roll_days_after"] = int((~in_train).sum())

    report = (
        roll_day_report(open_, close, rdays, threshold=0.15, split=train_end)
        if rule else None
    )
    return PreparedSeries(
        returns=returns,
        proxies=built,
        floors=floors,
        roll_days=rdays,
        roll_handling=roll_handling,
        roll_rule=rule,
        train_end=train_end,
        diagnostics=diagnostics,
        roll_report=report,
    )
