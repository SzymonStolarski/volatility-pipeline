"""
Normality and serial-dependence diagnostics for daily return series.

Motivation (referee point): the Kolmogorov-Smirnov normality test applied to
raw daily returns assumes the observations are i.i.d. Daily returns are close to
serially uncorrelated in the mean (white noise) but are NOT independent — they
exhibit volatility clustering (strong dependence in the squared/absolute
series). Under that dependence the standard KS / Jarque-Bere null distributions
are wrong: the effective sample size is below n, so the tests over-reject
(nominal p-values are too small). Plugging in sample mean/variance further
invalidates the standard KS critical values (Lilliefors).

This module addresses that in three layers:

Layer 1 — `dependence_diagnostics`: documents the dependence directly
    (Ljung-Box on returns and on squared returns, plus Engle's ARCH-LM). This
    turns the objection into motivating evidence: mean ~ white noise, variance
    strongly dependent -> conditional-variance (GARCH) modelling is warranted.

Layer 2 — `bai_ng_normality`: a normality test valid under serial dependence
    (Bai & Ng 2005, JBES 23(1)). It replaces the i.i.d. variances of the
    skewness/kurtosis statistics (6/T and 24/T, as used by Jarque-Bera) with HAC
    (Newey-West) long-run variances of the corresponding Hermite influence
    functions. Reduces exactly to Jarque-Bera when there is no serial
    dependence, so the classical JB is reported alongside for contrast.

Layer 3 — `ks_block_bootstrap`: keeps the KS statistic but replaces its i.i.d.
    p-value with a dependence-robust one from a stationary (block) bootstrap
    (Politis & Romano 1994), the same resampling scheme used by the MCS module.

`normality_report` runs all three and returns notebook-ready tables.

Dependencies: statsmodels (Ljung-Box, ARCH-LM), scipy (KS, Anderson-Darling).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm, chi2, anderson


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _hac_lrv(g: np.ndarray, lags: int) -> float:
    """
    Newey-West (Bartlett-kernel) long-run variance of a scalar series `g`.

    LRV = gamma_0 + 2 * sum_{j=1}^{lags} (1 - j/(lags+1)) * gamma_j,
    where gamma_j is the sample autocovariance at lag j (about the mean).
    Floored at a small positive number for numerical safety.
    """
    g = np.asarray(g, dtype=float)
    g = g - g.mean()
    T = g.size
    gamma0 = float(g @ g) / T
    s = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        gamma_j = float(g[j:] @ g[:-j]) / T
        s += 2.0 * w * gamma_j
    return max(s, 1e-12)


def _default_hac_lags(T: int) -> int:
    """Newey-West automatic bandwidth: floor(4 * (T/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))


def _stationary_block_indices(
    T: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Stationary-bootstrap index vector (Politis & Romano 1994): geometrically
    distributed block lengths (mean = block_size) with circular wrap-around.
    Mirrors the resampling in `mcs._stationary_bootstrap_means`.
    """
    p = 1.0 / block_size
    idx = np.empty(T, dtype=np.intp)
    pos = 0
    while pos < T:
        start = int(rng.integers(T))
        length = min(int(rng.geometric(p)), T - pos)
        idx[pos: pos + length] = (start + np.arange(length)) % T
        pos += length
    return idx


# ---------------------------------------------------------------------------
# Layer 1 — serial-dependence diagnostics
# ---------------------------------------------------------------------------

def dependence_diagnostics(
    returns,
    lb_lags: tuple[int, ...] = (10, 20),
    arch_lags: int = 10,
) -> pd.DataFrame:
    """
    Document serial dependence in returns.

    - Ljung-Box on returns          -> tests autocorrelation in the MEAN
      (expected: not significant; returns are ~ white noise).
    - Ljung-Box on squared returns  -> tests autocorrelation in the VARIANCE
      (expected: strongly significant; volatility clustering).
    - Engle ARCH-LM on returns      -> tests for ARCH effects
      (expected: strongly significant).

    Returns a tidy DataFrame: test | lag | statistic | p_value.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    r = pd.Series(returns, dtype=float).dropna()
    rows: list[dict] = []

    lb_r = acorr_ljungbox(r, lags=list(lb_lags), return_df=True)
    for lag, row in lb_r.iterrows():
        rows.append({"test": "Ljung-Box (returns)", "lag": int(lag),
                     "statistic": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])})

    lb_r2 = acorr_ljungbox(r ** 2, lags=list(lb_lags), return_df=True)
    for lag, row in lb_r2.iterrows():
        rows.append({"test": "Ljung-Box (squared returns)", "lag": int(lag),
                     "statistic": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])})

    lm_stat, lm_pval, _f_stat, _f_pval = het_arch(r, nlags=arch_lags)
    rows.append({"test": "Engle ARCH-LM", "lag": int(arch_lags),
                 "statistic": float(lm_stat), "p_value": float(lm_pval)})

    return pd.DataFrame(rows, columns=["test", "lag", "statistic", "p_value"])


# ---------------------------------------------------------------------------
# Layer 2 — Bai & Ng (2005) dependence-robust normality test
# ---------------------------------------------------------------------------

def bai_ng_normality(x, hac_lags: int | None = None) -> dict:
    """
    Bai & Ng (2005) skewness, kurtosis and joint normality tests, robust to
    serial dependence.

    The standardized series z_t = (x_t - mean) / sd has sample skewness
    E[z^3] and excess kurtosis E[z^4] - 3. Their asymptotic variances under H0
    are the long-run variances of the Hermite influence functions
        h3(z) = z^3 - 3 z      (skewness),
        h4(z) = z^4 - 6 z^2 + 3 (kurtosis),
    estimated here with a Newey-West HAC estimator. Test statistics:
        S = sqrt(T) * skew        / sqrt(LRV(h3))   ~ N(0, 1),
        K = sqrt(T) * excess_kurt / sqrt(LRV(h4))   ~ N(0, 1),
        joint = S^2 + K^2                            ~ chi^2(2).
    Under i.i.d. normality LRV(h3) -> 6 and LRV(h4) -> 24, so the joint statistic
    reduces to Jarque-Bera; JB is reported alongside for contrast.

    Returns a dict of statistics and p-values (robust and i.i.d.).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if hac_lags is None:
        hac_lags = _default_hac_lags(T)

    z = (x - x.mean()) / x.std(ddof=0)
    h3 = z ** 3 - 3.0 * z
    h4 = z ** 4 - 6.0 * z ** 2 + 3.0

    skew = float(np.mean(z ** 3))
    exkurt = float(np.mean(z ** 4) - 3.0)

    lrv3 = _hac_lrv(h3, hac_lags)
    lrv4 = _hac_lrv(h4, hac_lags)

    S = np.sqrt(T) * skew / np.sqrt(lrv3)
    K = np.sqrt(T) * exkurt / np.sqrt(lrv4)
    joint = float(S ** 2 + K ** 2)

    # i.i.d. Jarque-Bera counterpart (LRV fixed at 6 and 24)
    jb = float(T * skew ** 2 / 6.0 + T * exkurt ** 2 / 24.0)

    return {
        "n": T,
        "skewness": skew,
        "excess_kurtosis": exkurt,
        "hac_lags": int(hac_lags),
        "lrv_skew": float(lrv3),          # -> 6 under i.i.d. normal
        "lrv_kurt": float(lrv4),          # -> 24 under i.i.d. normal
        "bai_ng_skew_stat": float(S),
        "bai_ng_skew_pval": float(2.0 * norm.sf(abs(S))),
        "bai_ng_kurt_stat": float(K),
        "bai_ng_kurt_pval": float(2.0 * norm.sf(abs(K))),
        "bai_ng_joint_stat": joint,
        "bai_ng_joint_pval": float(chi2.sf(joint, 2)),
        "jarque_bera_stat": jb,
        "jarque_bera_pval": float(chi2.sf(jb, 2)),
    }


# ---------------------------------------------------------------------------
# Layer 3 — block-bootstrap KS p-value (dependence-robust)
# ---------------------------------------------------------------------------

def ks_block_bootstrap(
    x,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int | None = 42,
) -> dict:
    """
    Kolmogorov-Smirnov goodness-of-fit against Normal(mean, sd), with a
    dependence-robust p-value from a stationary (block) bootstrap.

    The KS statistic D = sup_x |F_T(x) - Phi_{mu,sd}(x)| is computed as usual.
    Its i.i.d. p-value (scipy) is invalid under serial dependence. We instead
    approximate the null sampling distribution of D via a recentered stationary
    bootstrap: for each resample b (geometric blocks, preserving dependence) we
    compute D_b = sup_x |F_T^b(x) - F_T(x)| — the fluctuation of the empirical
    CDF around its own value, which under H0 (data ~ Normal, so F_T ~ Phi)
    approximates the fluctuation of D. The bootstrap p-value is the share of
    D_b >= D_obs.

    Anderson-Darling (tail-weighted, more sensitive to fat tails than KS) is
    reported for reference with its 5% critical value.

    Caveat: the recentered bootstrap targets the dependence-driven fluctuation
    of the empirical process; it does not additionally correct for the
    estimation of (mu, sd) (the Lilliefors effect), which the classical i.i.d.
    KS p-value also ignores. For a fully specified null on the residual scale,
    prefer testing standardized GARCH residuals with a parametric bootstrap.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    mu = x.mean()
    sd = x.std(ddof=1)

    d_obs, p_iid = kstest(x, "norm", args=(mu, sd))

    if block_size is None:
        block_size = max(1, int(round(T ** (1.0 / 3.0))))

    rng = np.random.default_rng(seed)
    xs = np.sort(x)
    f_orig = np.arange(1, T + 1) / T  # empirical CDF of the original at xs

    d_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_block_indices(T, block_size, rng)
        xb_sorted = np.sort(x[idx])
        # empirical CDF of the resample evaluated at the original sorted points
        f_b = np.searchsorted(xb_sorted, xs, side="right") / T
        d_boot[b] = np.max(np.abs(f_b - f_orig))

    p_block = float((np.sum(d_boot >= d_obs) + 1) / (n_boot + 1))

    with warnings.catch_warnings():
        # SciPy >=1.17 warns that a `method` will become mandatory; we use the
        # classic tabulated critical values (index 2 == 5%), still supported.
        warnings.simplefilter("ignore", FutureWarning)
        ad = anderson(x, dist="norm")
    # scipy significance_level order is [15, 10, 5, 2.5, 1]; index 2 == 5%
    ad_crit_5 = float(ad.critical_values[2])

    return {
        "ks_stat": float(d_obs),
        "ks_pval_iid": float(p_iid),
        "ks_pval_block_boot": p_block,
        "block_size": int(block_size),
        "n_boot": int(n_boot),
        "ad_stat": float(ad.statistic),
        "ad_crit_5pct": ad_crit_5,
        "ad_reject_5pct": bool(ad.statistic > ad_crit_5),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper for the notebook
# ---------------------------------------------------------------------------

def normality_report(
    returns,
    *,
    name: str = "returns",
    lb_lags: tuple[int, ...] = (10, 20),
    arch_lags: int = 10,
    hac_lags: int | None = None,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """
    Run all three layers and return notebook-ready tables.

    Returns a dict with:
      - "dependence": Layer 1 table (Ljung-Box x2, ARCH-LM).
      - "normality" : combined table contrasting the i.i.d.-based tests
        (Jarque-Bera, classical KS) with the dependence-robust ones
        (Bai-Ng, block-bootstrap KS), plus Anderson-Darling.
    """
    dep = dependence_diagnostics(returns, lb_lags=lb_lags, arch_lags=arch_lags)
    dep.insert(0, "series", name)

    bn = bai_ng_normality(returns, hac_lags=hac_lags)
    ks = ks_block_bootstrap(returns, n_boot=n_boot, block_size=block_size, seed=seed)

    norm_rows = [
        {"test": "Jarque-Bera", "assumption": "i.i.d. (invalid here)",
         "statistic": bn["jarque_bera_stat"], "p_value": bn["jarque_bera_pval"]},
        {"test": "Bai-Ng joint (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_joint_stat"], "p_value": bn["bai_ng_joint_pval"]},
        {"test": "Bai-Ng skewness (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_skew_stat"], "p_value": bn["bai_ng_skew_pval"]},
        {"test": "Bai-Ng kurtosis (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_kurt_stat"], "p_value": bn["bai_ng_kurt_pval"]},
        {"test": "KS vs Normal", "assumption": "i.i.d. (invalid here)",
         "statistic": ks["ks_stat"], "p_value": ks["ks_pval_iid"]},
        {"test": "KS vs Normal (block-bootstrap)", "assumption": "dependence-robust",
         "statistic": ks["ks_stat"], "p_value": ks["ks_pval_block_boot"]},
        {"test": "Anderson-Darling", "assumption": f"i.i.d.; crit@5%={ks['ad_crit_5pct']:.3f}",
         "statistic": ks["ad_stat"], "p_value": np.nan},
    ]
    normality = pd.DataFrame(norm_rows, columns=["test", "assumption", "statistic", "p_value"])
    normality.insert(0, "series", name)

    return {"dependence": dep, "normality": normality}
