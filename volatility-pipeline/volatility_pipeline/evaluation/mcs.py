from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MCSResult:
    """Results from the Model Confidence Set procedure."""
    included: list[str]           # model names surviving in the MCS
    pvalues: dict[str, float]     # MCS p-value per model (> alpha ↔ in MCS)
    alpha: float
    loss: str

    def summary(self) -> pd.DataFrame:
        """
        DataFrame sorted by MCS p-value descending.
        Models with in_mcs=True belong to the MCS at level `alpha`.
        """
        df = (
            pd.DataFrame({"mcs_pvalue": self.pvalues})
            .sort_values("mcs_pvalue", ascending=False)
        )
        df["in_mcs"] = df["mcs_pvalue"] > self.alpha
        return df

    def __repr__(self) -> str:
        return (
            f"MCSResult(alpha={self.alpha}, loss={self.loss!r}, "
            f"n_included={len(self.included)}, models={self.included})"
        )


def mcs(
    results: dict,
    loss: str = "squared",
    alpha: float = 0.10,
    n_boot: int = 2000,
    block_size: int | None = None,
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
    block_size : stationary bootstrap block length (default: T^(1/3))
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

    if block_size is None:
        block_size = max(1, int(round(T ** (1.0 / 3.0))))

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
    )


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
