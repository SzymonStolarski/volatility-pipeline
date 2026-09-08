"""
Tests for the residual-level normality diagnostics (Layer 4 of normality.py).

The methodological defect these guard against: testing RAW RETURNS for
normality says nothing about which innovation distribution a GARCH model needs.
r_t = sqrt(h_t) * z_t is a variance mixture, so it is fat-tailed whenever h_t
varies — even when every z_t is exactly Gaussian. The distributional assumption
is about z_t, so the tests must be applied there, and the probability integral
transform must be taken under the distribution each model was actually fitted
with.

The discriminating test below is `test_pit_separates_right_from_wrong_dist`:
on data generated with t innovations, a Normal-innovation fit must be rejected
and a t-innovation fit must not.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.evaluation.normality import (
    dependence_diagnostics,
    residual_diagnostics_table,
    residual_report,
    uniformity_block_bootstrap,
)
from volatility_pipeline.models.garch_models import GARCHModel

N_BOOT = 400          # enough to resolve p ~ 0.05; keeps the suite quick
SEED = 11


def _simulate_garch(n, innovations, seed):
    """GARCH(1,1) returns driven by unit-variance `innovations`."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2.0e-6, 0.08, 0.90
    e = innovations(rng, n)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * e[t]
    idx = pd.period_range("2010-01-01", periods=n, freq="D")
    return pd.Series(r, index=idx)


def _normal_innov(rng, n):
    return rng.standard_normal(n)


def _t4_innov(rng, n):
    nu = 4.0
    return rng.standard_t(df=nu, size=n) / np.sqrt(nu / (nu - 2))


@pytest.fixture(scope="module")
def normal_returns():
    return _simulate_garch(3000, _normal_innov, seed=SEED)


@pytest.fixture(scope="module")
def t_returns():
    return _simulate_garch(3000, _t4_innov, seed=SEED)


# --------------------------------------------------------------------------
# Model accessors
# --------------------------------------------------------------------------

def test_std_resid_is_scale_invariant(normal_returns):
    """
    The `scale` factor appears in both the residual and the conditional
    volatility, so it must cancel. Anything that survives is optimiser noise,
    not a scale effect.
    """
    a = GARCHModel("GARCH", "normal", scale=100.0).fit(normal_returns).std_resid
    b = GARCHModel("GARCH", "normal", scale=1000.0).fit(normal_returns).std_resid
    assert np.max(np.abs(a.values - b.values)) < 1e-3


def test_std_resid_is_standardized(normal_returns):
    z = GARCHModel("GARCH", "normal").fit(normal_returns).std_resid
    assert len(z) == len(normal_returns)
    assert abs(float(z.mean())) < 0.1
    assert abs(float(z.std()) - 1.0) < 0.1


def test_dist_params_reports_shape_only(normal_returns):
    assert GARCHModel("GARCH", "normal").fit(normal_returns).dist_params == {}
    for dist in ("t", "ged"):
        p = GARCHModel("GARCH", dist).fit(normal_returns).dist_params
        assert list(p) == ["nu"] and p["nu"] > 0


def test_pit_lies_in_the_unit_interval(t_returns):
    for dist in ("normal", "t", "ged"):
        u = GARCHModel("GARCH", dist).fit(t_returns).pit()
        assert len(u) == len(t_returns)
        assert u.min() >= 0.0 and u.max() <= 1.0


def test_pit_uses_the_fitted_distribution_not_the_normal(t_returns):
    """A t fit and a Normal fit must not produce the same transform."""
    m_n = GARCHModel("GARCH", "normal").fit(t_returns)
    m_t = GARCHModel("GARCH", "t").fit(t_returns)
    assert np.max(np.abs(m_n.pit().values - m_t.pit().values)) > 0.01


def test_unfitted_model_raises():
    m = GARCHModel("GARCH", "normal")
    with pytest.raises(RuntimeError):
        _ = m.std_resid
    with pytest.raises(RuntimeError):
        m.pit()


# --------------------------------------------------------------------------
# The discriminating test
# --------------------------------------------------------------------------

def test_pit_separates_right_from_wrong_dist(t_returns):
    """
    On t-innovation data the PIT must reject the Normal fit and retain the t
    fit. This is the whole point of Layer 4: it is the only test that puts
    specifications with different assumed distributions on a common footing.
    """
    u_wrong = GARCHModel("GARCH", "normal").fit(t_returns).pit()
    u_right = GARCHModel("GARCH", "t").fit(t_returns).pit()

    wrong = uniformity_block_bootstrap(u_wrong, n_boot=N_BOOT, seed=SEED)
    right = uniformity_block_bootstrap(u_right, n_boot=N_BOOT, seed=SEED)

    assert wrong["ks_pval_block_boot"] < 0.05, "Normal fit on t data should be rejected"
    assert wrong["ad_reject_5pct"] is True
    assert right["ks_pval_block_boot"] > 0.05, "t fit on t data should be retained"
    assert right["ad_reject_5pct"] is False
    assert right["ad_stat"] < wrong["ad_stat"]


def test_correctly_specified_model_is_not_rejected(normal_returns):
    m = GARCHModel("GARCH", "normal").fit(normal_returns)
    rep = residual_report(m.std_resid, name="g", pit=m.pit(),
                          dist_label=m.dist, n_boot=N_BOOT, seed=SEED)

    robust = rep["normality"].query("assumption == 'dependence-robust'")
    assert (robust["p_value"] > 0.05).all()

    pit_ks = rep["pit"].query("test.str.contains('block-bootstrap')")["p_value"].iloc[0]
    assert pit_ks > 0.05


# --------------------------------------------------------------------------
# Variance-equation adequacy
# --------------------------------------------------------------------------

def test_garch_filtering_removes_the_arch_effect(normal_returns):
    """
    ARCH is present in the returns and must be gone from the standardized
    residuals — the precondition for reading anything into the distributional
    tests that follow.
    """
    raw = dependence_diagnostics(normal_returns, unit="returns")
    assert raw.loc[raw["test"] == "Engle ARCH-LM", "p_value"].iloc[0] < 0.01

    z = GARCHModel("GARCH", "normal").fit(normal_returns).std_resid
    res = dependence_diagnostics(z, unit="std. residuals")
    assert res.loc[res["test"] == "Engle ARCH-LM", "p_value"].iloc[0] > 0.05
    assert set(res["test"]) == {
        "Ljung-Box (std. residuals)",
        "Ljung-Box (squared std. residuals)",
        "Engle ARCH-LM",
    }


# --------------------------------------------------------------------------
# uniformity_block_bootstrap edge cases
# --------------------------------------------------------------------------

def test_uniformity_detects_a_non_uniform_sample():
    rng = np.random.default_rng(SEED)
    u = rng.beta(2.0, 5.0, size=1500)
    out = uniformity_block_bootstrap(u, n_boot=N_BOOT, seed=SEED)
    assert out["ks_pval_block_boot"] < 0.05
    assert out["ad_reject_5pct"] is True


def test_uniformity_retains_a_uniform_sample():
    rng = np.random.default_rng(SEED)
    out = uniformity_block_bootstrap(rng.uniform(size=1500), n_boot=N_BOOT, seed=SEED)
    assert out["ks_pval_block_boot"] > 0.05
    assert out["ad_reject_5pct"] is False


def test_anderson_darling_survives_a_saturated_pit():
    """
    A Normal CDF evaluated far into the tail saturates to exactly 0.0 or 1.0 in
    double precision. Without clipping, log(0) sends A^2 to -inf/inf for what is
    the single most informative observation in the sample.
    """
    rng = np.random.default_rng(SEED)
    u = rng.uniform(size=800)
    u[0], u[1] = 0.0, 1.0
    out = uniformity_block_bootstrap(u, n_boot=100, seed=SEED)
    assert np.isfinite(out["ad_stat"])


def test_uniformity_rejects_non_pit_input(normal_returns):
    """Passing standardized residuals instead of the PIT is a silent disaster;
    it must fail loudly instead."""
    z = GARCHModel("GARCH", "normal").fit(normal_returns).std_resid
    with pytest.raises(ValueError, match="probability-integral-transform"):
        uniformity_block_bootstrap(z, n_boot=50)


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------

def test_residual_diagnostics_table(normal_returns, t_returns):
    models = {
        "GARCH-NORMAL": GARCHModel("GARCH", "normal").fit(t_returns),
        "GARCH-T": GARCHModel("GARCH", "t").fit(t_returns),
    }
    tbl = residual_diagnostics_table(models, n_boot=N_BOOT, seed=SEED)

    assert list(tbl.index) == ["GARCH-NORMAL", "GARCH-T"]
    assert np.isnan(tbl.loc["GARCH-NORMAL", "nu"])
    assert tbl.loc["GARCH-T", "nu"] > 2
    # the misspecified row must be the one the PIT rejects
    assert tbl.loc["GARCH-NORMAL", "PIT KS p"] < 0.05
    assert tbl.loc["GARCH-T", "PIT KS p"] > 0.05
    # variance equation is fine in both
    assert (tbl["ARCH-LM p"] > 0.05).all()


def test_residual_diagnostics_table_duck_types_pit(normal_returns):
    """A model exposing only .std_resid must still work, with PIT columns NaN."""
    class _NoPit:
        dist = "normal"
        def __init__(self, z):
            self.std_resid = z

    z = GARCHModel("GARCH", "normal").fit(normal_returns).std_resid
    tbl = residual_diagnostics_table({"bare": _NoPit(z)}, n_boot=N_BOOT, seed=SEED)
    assert np.isnan(tbl.loc["bare", "PIT KS p"])
    assert np.isnan(tbl.loc["bare", "PIT AD"])
    assert not np.isnan(tbl.loc["bare", "BaiNg p"])
