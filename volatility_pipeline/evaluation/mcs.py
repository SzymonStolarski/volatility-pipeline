from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


def resolve_block_size(T: int, block_size=None) -> int:
    """
    Mean block length of the stationary bootstrap, stated explicitly.

      None or "sqrt"  int(sqrt(T)) — arch.bootstrap.MCS's own default, and so
                      what the arch p-values reported so far used (31 at T=1004)
      "cbrt"          round(T ** (1/3)) — the common rate-optimal rule (10)
      int             used as given

    Both MCS implementations take the block length from here, so they cannot
    silently use different ones. (Until October 2026 they did: this module
    defaulted to T^(1/3) while arch used sqrt(T).)
    """
    if block_size is None or block_size == "sqrt":
        return max(1, int(np.sqrt(T)))
    if block_size == "cbrt":
        return max(1, int(round(T ** (1.0 / 3.0))))
    if isinstance(block_size, (int, np.integer)) and block_size >= 1:
        return int(block_size)
    raise ValueError(f"block_size must be None, 'sqrt', 'cbrt' or a positive int, got {block_size!r}.")


@dataclass
class MCSResult:
    """Results from the Model Confidence Set procedure."""
    included: list[str]           # model names surviving in the MCS
    pvalues: dict[str, float]     # MCS p-value per model (> alpha ↔ in MCS)
    alpha: float
    loss: str
    block_size: int | None = None   # stationary-bootstrap mean block length used
    n_boot: int | None = None

    def summary(self, mark_early_stop: bool = False) -> pd.DataFrame:
        """
        DataFrame sorted by MCS p-value descending.
        Models with in_mcs=True belong to the MCS at level `alpha`.

        By default (`mark_early_stop=False`) the output is the original two-column
        frame (`mcs_pvalue`, `in_mcs`), where survivors carry the placeholder 1.0.

        This procedure stops at the first non-rejection (Hansen et al. 2011), so
        that 1.0 is *not* an actual bootstrap p-value — only a flag meaning "in
        the set at this alpha". Set `mark_early_stop=True` to instead show those
        placeholders as NaN in `mcs_pvalue` (so they are not misread as real
        p-values) and add a boolean `early_stop` column. For graded p-values that
        rank models within the set, use `arch_mcs()`.
        """
        df = (
            pd.DataFrame({"mcs_pvalue": self.pvalues})
            .sort_values("mcs_pvalue", ascending=False)
        )
        df["in_mcs"] = df["mcs_pvalue"] > self.alpha
        if mark_early_stop:
            df["early_stop"] = [name in self.included for name in df.index]
            df.loc[df["early_stop"], "mcs_pvalue"] = np.nan
        return df

    def __repr__(self) -> str:
        return (
            f"MCSResult(alpha={self.alpha}, loss={self.loss!r}, "
            f"block_size={self.block_size}, n_boot={self.n_boot}, "
            f"n_included={len(self.included)}, models={self.included})"
        )


def mcs(
    results: dict,
    loss: str = "squared",
    alpha: float = 0.10,
    n_boot: int = 2000,
    block_size: int | str | None = None,
    seed: int = 42,
) -> MCSResult:
    """
    Model Confidence Set (Hansen, Lunde & Nason 2011).

    Sequentially eliminates the worst-performing model until the null
    hypothesis of equal predictive accuracy cannot be rejected at level
    `alpha`. The surviving set is the (1-alpha) MCS.

    Uses the T_max statistic with stationary bootstrap (Politis & Romano 1994)
    for inference.

    Parameters
    ----------
    results    : dict mapping model name → ForecastResult
    loss       : loss function ('squared', 'absolute', 'qlike')
    alpha      : significance level (0.10 → 90% MCS; 0.25 → 75% MCS)
    n_boot     : bootstrap replications (≥ 1000 recommended)
    block_size : stationary bootstrap mean block length — see
                 resolve_block_size (default: int(sqrt(T)), as arch)
    seed       : random seed for reproducibility

    Returns
    -------
    MCSResult with .included, .pvalues, and .summary()
    """
    names = list(results.keys())
    m = len(names)

    loss_matrix = np.column_stack(
        [results[n].loss_series(loss).values for n in names]
    )  # shape: T × m
    T = loss_matrix.shape[0]

    block_size = resolve_block_size(T, block_size)

    rng = np.random.default_rng(seed)
    included_idx = list(range(m))
    pvalues_arr = np.zeros(m)
    prev_pval = 0.0

    while len(included_idx) > 1:
        L = loss_matrix[:, included_idx]  # T × k
        k = len(included_idx)

        # Loss differential relative to cross-sectional mean: d_{i.,t} = L_{it} - L_bar_t
        L_bar = L.mean(axis=1, keepdims=True)
        d = L - L_bar                     # T × k
        d_bar = d.mean(axis=0)            # k

        # Bootstrap variance of d_bar (stationary bootstrap under H0)
        boot_means = _stationary_bootstrap_means(d, block_size, n_boot, rng)  # n_boot × k
        var_boot = np.maximum(boot_means.var(axis=0, ddof=1), 1e-15)

        t_stats = d_bar / np.sqrt(var_boot)
        t_max_obs = float(t_stats.max())

        # Bootstrap T_max distribution: re-center so that E[boot_mean] = 0 under H0
        boot_t_max = ((boot_means - d_bar) / np.sqrt(var_boot)).max(axis=1)
        pval = float((boot_t_max >= t_max_obs).mean())

        # Enforce monotonicity: MCS p-values cannot decrease as set shrinks
        pval = max(pval, prev_pval)
        prev_pval = pval

        if pval > alpha:
            break  # H0 not rejected — remaining models form the MCS

        # Eliminate the model with the highest t-statistic (worst relative loss)
        worst_local = int(t_stats.argmax())
        worst_global = included_idx[worst_local]
        pvalues_arr[worst_global] = pval
        included_idx.pop(worst_local)

    # Surviving models receive p-value = 1.0
    for i in included_idx:
        pvalues_arr[i] = 1.0

    return MCSResult(
        included=[names[i] for i in included_idx],
        pvalues={names[i]: float(pvalues_arr[i]) for i in range(m)},
        alpha=alpha,
        loss=loss,
        block_size=block_size,
        n_boot=n_boot,
    )


def arch_mcs(
    results: dict,
    loss: str = "qlike",
    size: float = 0.10,
    n_boot: int = 2000,
    seed: int = 42,
    block_size: int | str | None = None,
) -> pd.DataFrame:
    """
    MCS p-values from Kevin Sheppard's arch.bootstrap.MCS (T_max statistic),
    stationary bootstrap — the implementation whose p-values the article
    reports.

    Run this alongside mcs() to distinguish implementation bugs from substantive
    results.  If both implementations retain the full model set, the issue is a
    noisy proxy (squared returns), not a code error.

    Returns a DataFrame with mcs_pvalue and in_mcs columns, compatible with
    MCSResult.summary() for direct side-by-side comparison.

    Parameters
    ----------
    results : dict mapping model name → ForecastResult
    loss    : loss function ('squared', 'absolute', 'qlike')
    size    : significance level (alpha)
    n_boot  : bootstrap replications
    seed    : random seed
    block_size : mean block length, see resolve_block_size. None keeps arch's
                 own default, int(sqrt(T)).

    The block length and replications used are recorded in `.attrs`.
    """
    from arch.bootstrap import MCS as ArchMCS

    names = list(results.keys())
    losses_df = pd.DataFrame(
        {n: results[n].loss_series(loss).values for n in names}
    )

    bs = resolve_block_size(len(losses_df), block_size)
    mcs_obj = ArchMCS(losses_df, size=size, reps=n_boot, block_size=bs,
                      method="max", seed=seed)
    mcs_obj.compute()

    pv = mcs_obj.pvalues
    if isinstance(pv, pd.DataFrame):
        pv = pv.iloc[:, 0]
    df = pd.DataFrame({"mcs_pvalue": pv})
    df["in_mcs"] = df["mcs_pvalue"] > size
    df = df.sort_values("mcs_pvalue", ascending=False)
    df.attrs.update(block_size=bs, n_boot=n_boot, size=size, loss=loss)
    return df


def _stationary_bootstrap_means(
    data: np.ndarray,
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary bootstrap (Politis & Romano 1994) sample means.

    Block lengths are geometrically distributed with mean = block_size.
    Returns array of shape (n_boot, k) where each row is the mean of
    one bootstrap resample.
    """
    T, k = data.shape
    p = 1.0 / block_size  # geometric parameter
    out = np.empty((n_boot, k))

    for b in range(n_boot):
        indices = np.empty(T, dtype=np.intp)
        pos = 0
        while pos < T:
            start = int(rng.integers(T))
            length = min(int(rng.geometric(p)), T - pos)
            src = (start + np.arange(length)) % T
            indices[pos: pos + length] = src
            pos += length
        out[b] = data[indices].mean(axis=0)

    return out
