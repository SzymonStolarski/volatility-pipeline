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

Layer 4 — `residual_report` / `residual_diagnostics_table`: the same battery
    applied to the STANDARDIZED RESIDUALS z_t = (r_t - mu) / sqrt(h_t) of a
    fitted conditional-variance model, plus a probability-integral-transform
    test of the model's own innovation distribution. Layers 1-3 characterise
    the unconditional distribution of the returns, which is a variance mixture
    and is fat-tailed under any GARCH model whatsoever — so it cannot support a
    choice between Normal, t and GED innovations. Layer 4 is what does.

`normality_report` runs Layers 1-3 on a return series; `residual_report` runs
Layer 4 on one fitted model; `residual_diagnostics_table` summarises Layer 4
across a whole set of them. All return notebook-ready tables.

MAIN AND SUPPLEMENTARY TESTS (article specification, October 2026). On the
returns, Bai-Ng is the main (descriptive) test. On the residuals, the main
battery is variance-equation adequacy (Ljung-Box on z and z^2, ARCH-LM) and
the PIT test of each model's own distribution, decided by Anderson-Darling
with a Monte Carlo p-value (`anderson_darling_uniform_test`). The
Kolmogorov-Smirnov tests and the Normal-null battery on the residuals are kept
as SUPPLEMENTARY output: tables carry a `role` column, and
`residual_diagnostics_table` lists its main and supplementary columns in
`.attrs`.

Dependencies: statsmodels (Ljung-Box, ARCH-LM), scipy (KS, Anderson-Darling).
"""
from __future__ import annotations

import warnings
from functools import lru_cache

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


def _as_clean_series(x, min_obs: int, caller: str) -> pd.Series:
    """
    Coerce input to a 1-D float Series with NaNs dropped, and fail loudly if
    there are too few observations.

    Without this guard a short or empty series reaches statsmodels/numpy and
    surfaces as an opaque error ("negative dimensions are not allowed"), which
    is almost always an upstream data problem — e.g. a failed yfinance download
    leaving `returns` empty, or an over-narrow date filter.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"{caller}: expected a 1-D series, got a DataFrame with "
                f"{x.shape[1]} columns. Pass a single column (e.g. df['Close'])."
            )
        x = x.iloc[:, 0]
    s = pd.Series(np.asarray(x, dtype=float).ravel()).dropna()
    if s.size < min_obs:
        raise ValueError(
            f"{caller}: only {s.size} usable (non-NaN) observation(s); "
            f"at least {min_obs} are required. This usually means the input "
            f"series is empty or nearly empty upstream — check that the price "
            f"download succeeded and that the date filters are not too narrow "
            f"(e.g. print len(returns) before calling)."
        )
    return s


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
    unit: str = "returns",
) -> pd.DataFrame:
    """
    Document serial dependence in returns.

    - Ljung-Box on returns          -> tests autocorrelation in the MEAN
      (expected: not significant; returns are ~ white noise).
    - Ljung-Box on squared returns  -> tests autocorrelation in the VARIANCE
      (expected: strongly significant; volatility clustering).
    - Engle ARCH-LM on returns      -> tests for ARCH effects
      (expected: strongly significant).

    Lags larger than the sample allows are dropped/capped rather than passed
    through to statsmodels, which would fail with an opaque array-shape error.

    `unit` only names the series in the "test" labels. Set it to
    "std. residuals" when running this on standardized GARCH residuals, where
    the SAME three tests carry the opposite expectation: after a well-specified
    conditional-variance model there should be no autocorrelation left in the
    squared series and no remaining ARCH, so a rejection there is evidence the
    variance equation is misspecified rather than evidence for GARCH.

    Returns a tidy DataFrame: test | lag | statistic | p_value.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    r = _as_clean_series(returns, min_obs=30, caller="dependence_diagnostics")
    T = r.size
    rows: list[dict] = []

    # Ljung-Box needs lag < T; ARCH-LM regresses on `arch_lags` lags, so it needs
    # comfortably more observations than lags.
    lb_use = [int(l) for l in lb_lags if 0 < int(l) < T]
    if not lb_use:
        lb_use = [max(1, min(10, T - 1))]
    arch_use = max(1, min(int(arch_lags), (T - 1) // 3))

    lb_r = acorr_ljungbox(r, lags=lb_use, return_df=True)
    for lag, row in lb_r.iterrows():
        rows.append({"test": f"Ljung-Box ({unit})", "lag": int(lag),
                     "statistic": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])})

    lb_r2 = acorr_ljungbox(r ** 2, lags=lb_use, return_df=True)
    for lag, row in lb_r2.iterrows():
        rows.append({"test": f"Ljung-Box (squared {unit})", "lag": int(lag),
                     "statistic": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])})

    lm_stat, lm_pval, _f_stat, _f_pval = het_arch(r, nlags=arch_use)
    rows.append({"test": "Engle ARCH-LM", "lag": int(arch_use),
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
    x = _as_clean_series(x, min_obs=20, caller="bai_ng_normality").to_numpy()
    x = x[np.isfinite(x)]
    T = x.size
    if hac_lags is None:
        hac_lags = _default_hac_lags(T)
    hac_lags = max(1, min(int(hac_lags), T - 2))

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
    x = _as_clean_series(x, min_obs=20, caller="ks_block_bootstrap").to_numpy()
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

    # Bai-Ng is the main (descriptive) normality test on returns; Jarque-Bera,
    # KS and Anderson-Darling are kept as supplementary rows.
    norm_rows = [
        {"role": "main", "test": "Bai-Ng joint (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_joint_stat"], "p_value": bn["bai_ng_joint_pval"]},
        {"role": "main", "test": "Bai-Ng skewness (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_skew_stat"], "p_value": bn["bai_ng_skew_pval"]},
        {"role": "main", "test": "Bai-Ng kurtosis (HAC)", "assumption": "dependence-robust",
         "statistic": bn["bai_ng_kurt_stat"], "p_value": bn["bai_ng_kurt_pval"]},
        {"role": "supplementary", "test": "Jarque-Bera", "assumption": "i.i.d. (invalid here)",
         "statistic": bn["jarque_bera_stat"], "p_value": bn["jarque_bera_pval"]},
        {"role": "supplementary", "test": "KS vs Normal", "assumption": "i.i.d. (invalid here)",
         "statistic": ks["ks_stat"], "p_value": ks["ks_pval_iid"]},
        {"role": "supplementary", "test": "KS vs Normal (block-bootstrap)", "assumption": "dependence-robust",
         "statistic": ks["ks_stat"], "p_value": ks["ks_pval_block_boot"]},
        {"role": "supplementary", "test": "Anderson-Darling",
         "assumption": f"i.i.d.; crit@5%={ks['ad_crit_5pct']:.3f}",
         "statistic": ks["ad_stat"], "p_value": np.nan},
    ]
    normality = pd.DataFrame(norm_rows, columns=["role", "test", "assumption", "statistic", "p_value"])
    normality.insert(0, "series", name)

    return {"dependence": dep, "normality": normality}


# ---------------------------------------------------------------------------
# Layer 4 — the same battery on standardized GARCH residuals
# ---------------------------------------------------------------------------
#
# Why this layer exists. Layers 1-3 describe the UNCONDITIONAL distribution of
# the returns. That distribution is a variance mixture: even if every
# innovation z_t is exactly Gaussian, r_t = sqrt(h_t) * z_t is fat-tailed
# whenever h_t varies. Rejecting normality on raw returns is therefore expected
# under any GARCH model and carries no information about which innovation
# distribution to use. The quantity the `dist` choice is actually about is
# z_t = (r_t - mu) / sqrt(h_t), and that is what this layer tests.
#
# Two distinct questions are asked, and they should not be conflated:
#
#   (a) Is the NORMAL innovation assumption adequate?  -> test z_t for
#       normality (Bai-Ng, block-bootstrap KS, Anderson-Darling). Applied to a
#       Normal-GARCH fit this is the direct evidence for or against needing a
#       fat-tailed innovation.
#
#   (b) Is the FITTED distribution adequate, whichever it is? -> probability
#       integral transform u_t = F(z_t; theta_hat) under the model's own
#       conditional distribution, tested for uniformity (Diebold, Gunther & Tay
#       1998). This is the only test that puts Normal, t and GED specifications
#       on a common footing, since each is judged against the distribution it
#       was estimated under.
#
# `residual_report` runs both, plus the Layer-1 dependence battery re-purposed
# as a variance-equation adequacy check (after a well-specified model there
# should be no ARCH left in z_t).

# Asymptotic Anderson-Darling critical value at 5% for a FULLY SPECIFIED
# continuous null (Marsaglia & Marsaglia 2004). Kept for reference only: the PIT
# verdict now uses the Monte Carlo p-value of `anderson_darling_uniform_test`,
# whose simulated 5% cut-off for n = 3110 is 2.495.
_AD_CRIT_5PCT_FULLY_SPECIFIED = 2.492


def _anderson_darling_uniform_rows(U: np.ndarray) -> np.ndarray:
    """
    Anderson-Darling statistic for the U(0, 1) null, one per ROW of U.

    A^2 = -n - (1/n) * sum_i (2i-1) * [ln u_(i) + ln(1 - u_(n+1-i))]

    Values are clipped away from 0 and 1 before the logs. This is not cosmetic:
    a Normal-innovation fit on fat-tailed data produces standardized residuals
    far enough into the tail that the normal CDF saturates to exactly 0.0 or
    1.0 in double precision, which would send A^2 to infinity for what is in
    fact the most informative observation. The Monte Carlo null below is built
    with this same function, so the p-value is exact for the clipped statistic.
    """
    U = np.sort(np.atleast_2d(np.asarray(U, dtype=float)), axis=1)
    n = U.shape[1]
    eps = 1.0 / (4.0 * n)      # tighter than the smallest resolvable order stat
    U = np.clip(U, eps, 1.0 - eps)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(U) + np.log1p(-U[:, ::-1])), axis=1)
    return -n - s / n


def _anderson_darling_uniform(u: np.ndarray) -> float:
    """Anderson-Darling statistic of one sample against U(0, 1)."""
    return float(_anderson_darling_uniform_rows(np.asarray(u, dtype=float)[None, :])[0])


@lru_cache(maxsize=32)
def _ad_uniform_null(n: int, n_mc: int, seed: int) -> np.ndarray:
    """
    Sorted Monte Carlo draws of A^2 for n i.i.d. U(0, 1) observations.

    Depends only on (n, n_mc, seed), so it is computed once and shared by every
    model fitted on the same window. Checked against the asymptotic critical
    values of the fully specified case (1.933 / 2.492 / 3.857 at 10/5/1%):
    the n = 3110 draws give 1.95 / 2.495 / 3.79.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_mc)
    chunk = max(1, min(n_mc, 4_000_000 // max(n, 1)))
    for start in range(0, n_mc, chunk):
        stop = min(n_mc, start + chunk)
        out[start:stop] = _anderson_darling_uniform_rows(rng.random((stop - start, n)))
    out.sort()
    return out


def anderson_darling_uniform_test(u, n_mc: int = 10_000, seed: int = 20261001) -> dict:
    """
    The main PIT test: Anderson-Darling against U(0, 1), with a Monte Carlo
    p-value.

    Why Anderson-Darling. The innovation distribution is decided in the tails,
    and A^2 weights departures there; KS is most sensitive near the median and
    can miss a distribution that is wrong only in the tails. KS is kept as a
    supplementary test (`uniformity_block_bootstrap`).

    Why this reference distribution. Under correct specification of both the
    variance equation and the innovation distribution, u_t = F(z_t; theta_hat)
    is i.i.d. U(0, 1) (Diebold, Gunther & Tay 1998), so the null distribution
    of A^2 is that of a fully specified uniform sample, simulated here. Serial
    dependence is not part of this null; it is tested separately by the
    Ljung-Box and ARCH-LM columns. The shape parameter of t is estimated, which
    makes this reference slightly conservative (estimation shrinks the true
    null distribution of A^2), so a rejection is if anything understated.

    p_value = (1 + #{simulated A^2 >= observed}) / (n_mc + 1); its floor is
    1 / (n_mc + 1).
    """
    u = _as_clean_series(u, min_obs=20, caller="anderson_darling_uniform_test").to_numpy()
    u = u[np.isfinite(u)]
    if not ((u >= 0.0).all() and (u <= 1.0).all()):
        raise ValueError(
            "anderson_darling_uniform_test expects probability-integral-transform "
            "values in [0, 1]; pass model.pit(), not the residuals themselves."
        )
    n = int(u.size)
    stat = _anderson_darling_uniform(u)
    null = _ad_uniform_null(n, int(n_mc), int(seed))
    n_ge = n_mc - int(np.searchsorted(null, stat, side="left"))
    pval = (1 + n_ge) / (n_mc + 1)
    return {
        "n": n,
        "ad_stat": float(stat),
        "ad_pval": float(pval),
        "ad_crit_5pct": float(np.quantile(null, 0.95)),
        "n_mc": int(n_mc),
    }


def uniformity_block_bootstrap(
    u,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int | None = 42,
    n_mc: int = 10_000,
) -> dict:
    """
    Test probability-integral-transform values for uniformity on (0, 1), with a
    dependence-robust p-value from a stationary (block) bootstrap.

    Under correct specification of both the variance equation and the
    innovation distribution, u_t = F(z_t; theta_hat) is i.i.d. U(0, 1). The KS
    statistic D = sup_x |F_T(x) - x| measures the departure. Its i.i.d. p-value
    is reported but is optimistic on two counts — any dependence left in z_t,
    and the estimation of theta_hat — so the bootstrap p-value, obtained from
    the same recentered stationary-bootstrap construction used by
    `ks_block_bootstrap`, is the one to report.

    Anderson-Darling is the MAIN verdict (see `anderson_darling_uniform_test`
    for its p-value): it weights the tails, which is precisely where a Normal
    innovation assumption fails and where the choice between Normal and t is
    decided; KS is most sensitive near the median and can miss it. The KS
    results are returned as supplementary evidence.
    """
    u = _as_clean_series(u, min_obs=20, caller="uniformity_block_bootstrap").to_numpy()
    u = u[np.isfinite(u)]
    T = u.size
    if not ((u >= 0.0).all() and (u <= 1.0).all()):
        raise ValueError(
            "uniformity_block_bootstrap expects probability-integral-transform "
            "values in [0, 1]; got values outside that range. Pass model.pit(), "
            "not the standardized residuals themselves."
        )

    d_obs, p_iid = kstest(u, "uniform")

    if block_size is None:
        block_size = max(1, int(round(T ** (1.0 / 3.0))))

    rng = np.random.default_rng(seed)
    us = np.sort(u)
    f_orig = np.arange(1, T + 1) / T

    d_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_block_indices(T, block_size, rng)
        ub_sorted = np.sort(u[idx])
        f_b = np.searchsorted(ub_sorted, us, side="right") / T
        d_boot[b] = np.max(np.abs(f_b - f_orig))

    p_block = float((np.sum(d_boot >= d_obs) + 1) / (n_boot + 1))
    ad = anderson_darling_uniform_test(u, n_mc=n_mc)

    return {
        "n": int(T),
        "ad_stat": ad["ad_stat"],
        "ad_pval": ad["ad_pval"],
        "ad_crit_5pct": ad["ad_crit_5pct"],
        "ad_reject_5pct": bool(ad["ad_pval"] < 0.05),
        "ad_n_mc": ad["n_mc"],
        "ks_stat": float(d_obs),
        "ks_pval_iid": float(p_iid),
        "ks_pval_block_boot": p_block,
        "block_size": int(block_size),
        "n_boot": int(n_boot),
    }


def residual_report(
    std_resid,
    *,
    name: str = "model",
    pit=None,
    dist_label: str | None = None,
    lb_lags: tuple[int, ...] = (10, 20),
    arch_lags: int = 10,
    hac_lags: int | None = None,
    n_boot: int = 2000,
    block_size: int | None = None,
    seed: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """
    Run the diagnostic battery on one model's standardized residuals.

    Parameters
    ----------
    std_resid  : z_t from a fitted conditional-variance model (GARCHModel.std_resid).
    name       : model label, copied into every returned table.
    pit        : optional u_t = F(z_t; theta_hat) (GARCHModel.pit()). When given,
                 a third table tests the FITTED distribution; when omitted only
                 the Normal null is examined.
    dist_label : name of the fitted innovation distribution, for the PIT table's
                 H0 column (e.g. "t", "ged").

    Returns
    -------
    dict with:
      "adequacy"  — Ljung-Box on z and z^2 plus ARCH-LM. Here a NON-rejection is
                    the good outcome: it means the variance equation has removed
                    the clustering. Rejection indicates a misspecified h_t, and
                    any distributional conclusion drawn below it is unsafe.
      "normality" — is the NORMAL innovation assumption adequate for z_t?
      "pit"       — is the model's OWN innovation distribution adequate?
                    Absent from the dict when `pit` is not supplied.
    """
    z = _as_clean_series(std_resid, min_obs=30, caller="residual_report")

    adequacy = dependence_diagnostics(
        z, lb_lags=lb_lags, arch_lags=arch_lags, unit="std. residuals"
    )
    adequacy.insert(0, "role", "main")
    adequacy.insert(0, "model", name)

    bn = bai_ng_normality(z, hac_lags=hac_lags)
    ks = ks_block_bootstrap(z, n_boot=n_boot, block_size=block_size, seed=seed)
    # On the residuals the whole Normal-null battery is supplementary: the PIT
    # below is the test that judges each model by its own distribution.
    normality = pd.DataFrame(
        [
            {"test": "Jarque-Bera", "assumption": "i.i.d.",
             "statistic": bn["jarque_bera_stat"], "p_value": bn["jarque_bera_pval"]},
            {"test": "Bai-Ng joint (HAC)", "assumption": "dependence-robust",
             "statistic": bn["bai_ng_joint_stat"], "p_value": bn["bai_ng_joint_pval"]},
            {"test": "Bai-Ng skewness (HAC)", "assumption": "dependence-robust",
             "statistic": bn["bai_ng_skew_stat"], "p_value": bn["bai_ng_skew_pval"]},
            {"test": "Bai-Ng kurtosis (HAC)", "assumption": "dependence-robust",
             "statistic": bn["bai_ng_kurt_stat"], "p_value": bn["bai_ng_kurt_pval"]},
            {"test": "KS vs Normal", "assumption": "i.i.d.",
             "statistic": ks["ks_stat"], "p_value": ks["ks_pval_iid"]},
            {"test": "KS vs Normal (block-bootstrap)", "assumption": "dependence-robust",
             "statistic": ks["ks_stat"], "p_value": ks["ks_pval_block_boot"]},
            {"test": "Anderson-Darling vs Normal",
             "assumption": f"i.i.d.; crit@5%={ks['ad_crit_5pct']:.3f}",
             "statistic": ks["ad_stat"], "p_value": np.nan},
        ],
        columns=["test", "assumption", "statistic", "p_value"],
    )
    normality.insert(0, "role", "supplementary")
    normality.insert(0, "model", name)

    out = {"adequacy": adequacy, "normality": normality}

    if pit is not None:
        uni = uniformity_block_bootstrap(
            pit, n_boot=n_boot, block_size=block_size, seed=seed
        )
        h0 = f"fitted {dist_label}" if dist_label else "fitted distribution"
        pit_tbl = pd.DataFrame(
            [
                {"role": "main", "test": "Anderson-Darling vs Uniform(0,1)",
                 "assumption": f"Monte Carlo p ({uni['ad_n_mc']} draws); "
                               f"crit@5%={uni['ad_crit_5pct']:.3f}; conservative",
                 "statistic": uni["ad_stat"], "p_value": uni["ad_pval"]},
                {"role": "supplementary", "test": "KS vs Uniform(0,1) (block-bootstrap)",
                 "assumption": "dependence-robust",
                 "statistic": uni["ks_stat"], "p_value": uni["ks_pval_block_boot"]},
                {"role": "supplementary", "test": "KS vs Uniform(0,1)", "assumption": "i.i.d.",
                 "statistic": uni["ks_stat"], "p_value": uni["ks_pval_iid"]},
            ],
            columns=["role", "test", "assumption", "statistic", "p_value"],
        )
        pit_tbl.insert(0, "H0", h0)
        pit_tbl.insert(0, "model", name)
        out["pit"] = pit_tbl

    return out


def residual_diagnostics_table(
    models: dict,
    *,
    lb_lag: int = 20,
    arch_lags: int = 10,
    hac_lags: int | None = None,
    n_boot: int = 1000,
    block_size: int | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    One-row-per-model summary across a set of fitted conditional-variance models
    — the publication-facing version of `residual_report`.

    `models` maps a display name to any object exposing `.std_resid` and,
    optionally, `.pit()`, `.dist` and `.dist_params` (GARCHModel does). Nothing
    is imported from the models package: the contract is duck-typed, exactly as
    RollingEvaluator's is, so this module stays free of model dependencies.

    Columns
    -------
    MAIN (listed in `.attrs["main_columns"]`):
    dist, nu, n               the fitted innovation distribution, its shape
                              parameter (blank for Normal) and the sample size.
    LB(lag) p                 Ljung-Box on z_t: is the mean equation adequate?
    LB2(lag) p, ARCH-LM p     variance-equation adequacy. LARGE p is the good
                              outcome: no clustering left in z_t.
    PIT AD, PIT AD p          is the model's OWN distribution adequate?
                              Anderson-Darling on u_t = F(z_t), Monte Carlo
                              p-value. LARGE p is the good outcome. NaN when the
                              model does not expose .pit().

    SUPPLEMENTARY (`.attrs["supplementary_columns"]`):
    skew, ex.kurt             shape of the standardized residuals.
    BaiNg p, KS-N p           is the NORMAL innovation assumption adequate?
                              SMALL p rejects it.
    AD-N                      Anderson-Darling against Normal (statistic).
    PIT KS p                  KS on u_t, block-bootstrap p-value.

    `n_boot` defaults lower than elsewhere in this module because two block
    bootstraps run per model; raise it for final numbers.
    """
    rows: list[dict] = []
    for name, m in models.items():
        z = _as_clean_series(m.std_resid, min_obs=30, caller="residual_diagnostics_table")
        T = z.size
        lag = max(1, min(int(lb_lag), T - 1))

        dep = dependence_diagnostics(
            z, lb_lags=(lag,), arch_lags=arch_lags, unit="std. residuals"
        )
        lb_p = float(dep.loc[dep["test"] == "Ljung-Box (std. residuals)", "p_value"].iloc[0])
        lb2_p = float(dep.loc[dep["test"].str.startswith("Ljung-Box (squared"), "p_value"].iloc[0])
        arch_p = float(dep.loc[dep["test"] == "Engle ARCH-LM", "p_value"].iloc[0])

        bn = bai_ng_normality(z, hac_lags=hac_lags)
        ks = ks_block_bootstrap(z, n_boot=n_boot, block_size=block_size, seed=seed)

        shape = getattr(m, "dist_params", {}) or {}
        row = {
            "model": name,
            "dist": str(getattr(m, "dist", "")),
            "nu": float(next(iter(shape.values()))) if shape else np.nan,
            "n": T,
            f"LB({lag}) p": lb_p,
            f"LB2({lag}) p": lb2_p,
            "ARCH-LM p": arch_p,
            "skew": bn["skewness"],
            "ex.kurt": bn["excess_kurtosis"],
            "BaiNg p": bn["bai_ng_joint_pval"],
            "KS-N p": ks["ks_pval_block_boot"],
            "AD-N": ks["ad_stat"],
        }

        pit_fn = getattr(m, "pit", None)
        if callable(pit_fn):
            uni = uniformity_block_bootstrap(
                pit_fn(), n_boot=n_boot, block_size=block_size, seed=seed
            )
            row["PIT AD"] = uni["ad_stat"]
            row["PIT AD p"] = uni["ad_pval"]
            row["PIT KS p"] = uni["ks_pval_block_boot"]
        else:
            row["PIT AD"] = row["PIT AD p"] = row["PIT KS p"] = np.nan

        rows.append(row)

    main = ["dist", "nu", "n", f"LB({lag}) p", f"LB2({lag}) p", "ARCH-LM p", "PIT AD", "PIT AD p"]
    supplementary = ["skew", "ex.kurt", "BaiNg p", "KS-N p", "AD-N", "PIT KS p"]
    out = pd.DataFrame(rows).set_index("model")[main + supplementary]
    out.attrs["main_columns"] = main
    out.attrs["supplementary_columns"] = supplementary
    return out
