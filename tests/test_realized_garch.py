"""
Tests for the Realized GARCH implementation.

This model is written from scratch — `arch` has no Realized GARCH — so the
estimator itself has to be validated, not just its plumbing. The only honest way
to check a hand-written joint MLE is to simulate from the model with KNOWN
parameters and confirm they come back; `test_recovers_simulated_parameters` and
`test_recovers_the_latent_variance` do that.

Two failure modes these guard in particular:

1. A units mismatch in the likelihood. A log-likelihood is not scale-free, so a
   RealizedGARCH fitted on raw returns and a GARCHModel fitted on percent
   returns differ by n*log(100) — about 14,300 on a 3,110-day sample — for
   reasons having nothing to do with fit. In a joint information-criteria table
   that would make this model look unbeatable by an enormous margin, silently.
   `test_likelihood_scale_shift_is_exact` pins the convention.

2. Look-ahead through the realized measure. The model holds the whole series and
   reindexes it onto the returns it is given, so a slicing mistake would let it
   condition on the future without any error being raised.
   `test_no_look_ahead_into_the_realized_measure` pins it.
"""
import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.evaluation.rolling_forecast import RollingEvaluator
from volatility_pipeline.models.realized_garch import RealizedGARCH

TRUE = {
    "omega": -0.397, "beta": 0.60, "gamma": 0.35, "xi": -0.18,
    "phi": 1.00, "tau1": -0.07, "tau2": 0.07, "sigma_u": 0.38,
}
TRUE_PERSISTENCE = TRUE["beta"] + TRUE["phi"] * TRUE["gamma"]


def simulate_realized_garch(n=6000, seed=1):
    """Simulate the model itself, returning returns, realized measure, true h."""
    rng = np.random.default_rng(seed)
    o, b, g = TRUE["omega"], TRUE["beta"], TRUE["gamma"]
    xi, phi = TRUE["xi"], TRUE["phi"]
    t1, t2, su = TRUE["tau1"], TRUE["tau2"], TRUE["sigma_u"]

    log_h = (o + g * xi) / (1.0 - b - phi * g)
    r, x, h_true = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        z = rng.standard_normal()
        h = np.exp(log_h)
        h_true[t] = h
        r[t] = np.sqrt(h) * z
        log_x = xi + phi * log_h + (t1 * z + t2 * (z * z - 1.0)) + rng.standard_normal() * su
        x[t] = np.exp(log_x)
        log_h = o + b * log_h + g * log_x

    idx = pd.date_range("2005-01-03", periods=n, freq="B")
    return (pd.Series(r, index=idx),
            pd.Series(x, index=idx, name="sim_rm"),
            pd.Series(h_true, index=idx))


@pytest.fixture(scope="module")
def sim():
    return simulate_realized_garch()


# --------------------------------------------------------------------------
# The estimator itself
# --------------------------------------------------------------------------

def test_recovers_simulated_parameters(sim):
    """
    The dynamic parameters must come back. omega and xi are deliberately NOT
    asserted individually: a level shift c in log h is absorbed by
    omega -> omega + c(1-beta) and xi -> xi - phi*c, so only their combination
    is well identified. That combination is checked via the implied
    unconditional variance below.
    """
    r, x, _ = sim
    p = RealizedGARCH(x).fit(r).params
    assert p["beta"] == pytest.approx(TRUE["beta"], abs=0.05)
    assert p["gamma"] == pytest.approx(TRUE["gamma"], abs=0.05)
    assert p["phi"] == pytest.approx(TRUE["phi"], abs=0.10)
    assert p["tau1"] == pytest.approx(TRUE["tau1"], abs=0.03)
    assert p["tau2"] == pytest.approx(TRUE["tau2"], abs=0.03)
    assert p["sigma_u"] == pytest.approx(TRUE["sigma_u"], abs=0.03)


def test_recovers_the_persistence(sim):
    """Persistence is beta + phi*gamma, not beta alone."""
    r, x, _ = sim
    m = RealizedGARCH(x).fit(r)
    assert m.persistence == pytest.approx(TRUE_PERSISTENCE, abs=0.02)
    assert m.persistence == pytest.approx(
        m.params["beta"] + m.params["phi"] * m.params["gamma"], rel=1e-12
    )


def test_recovers_the_latent_variance(sim):
    """
    What the model is actually for: tracking the unobserved h_t. This is the
    check that matters even where individual parameters drift.
    """
    r, x, h_true = sim
    fitted = RealizedGARCH(x).fit(r).insample_variance()
    assert np.corrcoef(np.log(fitted.values), np.log(h_true.values))[0, 1] > 0.99
    assert float(fitted.mean() / h_true.mean()) == pytest.approx(1.0, abs=0.10)


def test_implied_unconditional_variance_is_right(sim):
    """omega and xi individually drift; the combination that sets the level
    must not."""
    r, x, _ = sim
    p = RealizedGARCH(x).fit(r).params
    implied = (p["omega"] + p["gamma"] * p["xi"]) / (
        1.0 - p["beta"] - p["phi"] * p["gamma"]
    )
    # on the scale=100 basis, so shift the truth by 2*log(100)
    true_implied = (TRUE["omega"] + TRUE["gamma"] * TRUE["xi"]) / (
        1.0 - TRUE_PERSISTENCE
    ) + 2.0 * np.log(100.0)
    assert implied == pytest.approx(true_implied, abs=0.5)


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

def test_one_step_forecast_matches_the_recursion(sim):
    """h_{T+1} = exp(omega + beta*log h_T + gamma*log x_T) — exact, because both
    inputs are observed at T."""
    r, x, _ = sim
    m = RealizedGARCH(x).fit(r)
    p = m.params
    log_h_T = float(np.log(m.insample_variance().iloc[-1] * m.scale ** 2))
    log_x_T = float(np.log(x.iloc[-1] * m.scale ** 2))
    manual = np.exp(p["omega"] + p["beta"] * log_h_T + p["gamma"] * log_x_T) / m.scale ** 2
    assert float(m.forecast_variance(1)[0]) == pytest.approx(manual, rel=1e-10)


def test_multistep_converges_to_the_unconditional(sim):
    r, x, _ = sim
    m = RealizedGARCH(x).fit(r)
    p = m.params
    fc = m.forecast_variance(400)
    uncond = np.exp(
        (p["omega"] + p["gamma"] * p["xi"]) / (1.0 - m.persistence)
    ) / m.scale ** 2
    assert fc[-1] == pytest.approx(uncond, rel=0.02)
    assert np.isfinite(fc).all() and (fc > 0).all()


# --------------------------------------------------------------------------
# Scale — the comparability fix
# --------------------------------------------------------------------------

def test_forecasts_are_scale_invariant(sim):
    r, x, _ = sim
    a = RealizedGARCH(x, scale=1.0).fit(r).forecast_variance(1)[0]
    b = RealizedGARCH(x, scale=100.0).fit(r).forecast_variance(1)[0]
    assert a == pytest.approx(b, rel=1e-3)


def test_likelihood_scale_shift_is_exact(sim):
    """
    The partial likelihood is -0.5*sum(log 2pi + log h + z^2); scaling returns by
    c sends log h -> log h + 2 log c, so the likelihood shifts by exactly
    -n*log(c). This is the whole reason `scale` defaults to 100 here: GARCHModel
    uses the same convention, and without it the two models' log-likelihoods
    would differ by ~14,300 on a 3,110-day sample for purely dimensional reasons.
    """
    r, x, _ = sim
    a = RealizedGARCH(x, scale=1.0).fit(r)
    b = RealizedGARCH(x, scale=100.0).fit(r)
    assert a.loglikelihood - b.loglikelihood == pytest.approx(
        len(r) * np.log(100.0), rel=1e-3
    )


def test_partial_and_joint_likelihoods_are_distinct(sim):
    """
    The joint figure scores the measurement equation too, so it is NOT
    comparable with a return-only GARCH likelihood. info_criteria() must expose
    the PARTIAL one under 'LogL', the key the rest of the pipeline compares on.
    """
    r, x, _ = sim
    m = RealizedGARCH(x).fit(r)
    ic = m.info_criteria()
    assert ic["LogL"] == m.loglikelihood
    assert m.joint_loglikelihood != pytest.approx(m.loglikelihood)
    assert ic["LogL_joint"] == m.joint_loglikelihood
    assert ic["BIC"] == pytest.approx(
        m.n_params * np.log(len(r)) - 2.0 * m.loglikelihood
    )


# --------------------------------------------------------------------------
# The realized measure is part of the specification
# --------------------------------------------------------------------------

def test_fit_ignores_the_target_argument(sim):
    """
    The realized measure comes from the constructor, so the notebook's ML
    training target must not be able to redefine this model.
    """
    r, x, _ = sim
    a = RealizedGARCH(x).fit(r)
    b = RealizedGARCH(x).fit(r, target=(r ** 2))
    pd.testing.assert_series_equal(a.params, b.params)


def test_different_measures_give_different_models(sim):
    r, x, _ = sim
    a = RealizedGARCH(x).fit(r)
    b = RealizedGARCH((r ** 2).rename("sq")).fit(r)
    assert a.persistence != pytest.approx(b.persistence, abs=1e-6)


def test_no_look_ahead_into_the_realized_measure(sim):
    """
    The model holds the whole series, so it must only ever read the dates
    present in the returns it is handed. Deleting everything after a cut point
    must leave forecasts up to that point untouched.
    """
    r, x, _ = sim
    cut = 4000
    full = RealizedGARCH(x).fit(r.iloc[:cut])
    truncated = RealizedGARCH(x.iloc[:cut]).fit(r.iloc[:cut])
    pd.testing.assert_series_equal(full.params, truncated.params)
    assert full.forecast_variance(1)[0] == pytest.approx(
        truncated.forecast_variance(1)[0], rel=1e-12
    )


def test_missing_dates_raise(sim):
    r, x, _ = sim
    with pytest.raises(ValueError, match="missing"):
        RealizedGARCH(x.iloc[:100]).fit(r)


def test_non_series_measure_raises(sim):
    r, x, _ = sim
    with pytest.raises(TypeError, match="pd.Series"):
        RealizedGARCH(x.values)


def test_non_positive_realized_values_are_floored(sim):
    """
    Garman-Klass is not bounded below by zero — BZ=F has 80 days at exactly 0
    and 7 negative — and log of those would poison the whole likelihood.
    """
    r, x, _ = sim
    x = x.copy()
    x.iloc[10] = 0.0
    x.iloc[20] = -1e-8
    m = RealizedGARCH(x).fit(r)
    assert np.isfinite(m.loglikelihood)
    assert np.isfinite(m.forecast_variance(1)[0])


def test_unfitted_model_raises(sim):
    _, x, _ = sim
    m = RealizedGARCH(x)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.forecast_variance(1)
    assert "unfitted" in repr(m)


# --------------------------------------------------------------------------
# Evaluator integration
# --------------------------------------------------------------------------

def test_works_inside_the_rolling_evaluator(sim):
    r, x, _ = sim
    train, test = r.iloc[:1200], r.iloc[1200:1400]
    res = RollingEvaluator(n_ahead=1, refit_every=10).evaluate(
        lambda: RealizedGARCH(x), "rg", train, test, actuals_series=x
    )
    f = res.forecasts.values
    assert len(np.unique(f)) == len(f), "forecasts frozen between re-fits"
    assert (f > 0).all() and np.isfinite(f).all()
