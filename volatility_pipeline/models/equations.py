"""
Reporting of estimated GARCH equations: coefficient tables and rendered
equations with the fitted numbers substituted in.

Motivation (referee point): the manuscript reported only information criteria
(LogL/AIC/BIC), not the estimated variance equations, so a reader cannot see
which coefficients are significant. In particular the referee asked whether the
GJR asymmetry parameter is redundant — a question that can only be answered from
the coefficient and its standard error.

Three levels of output:

- `param_table`   : tidy long table (model x parameter) with coefficient,
                    standard error, t-statistic, p-value and significance stars.
- `param_matrix`  : wide publication-style table, one row per model, cells
                    formatted "coef (std_err)" with stars — the layout usually
                    printed in papers.
- `equation_lines`/`equations_markdown` : the fitted mean and variance equations
                    per model, with estimated coefficients substituted, rendered
                    as LaTeX for display in a notebook.

Plus `asymmetry_summary`, which isolates the asymmetry coefficient of every
model that has one and reports whether it is statistically distinguishable from
zero — the direct answer to "is the asymmetry parameter redundant?".

Units
-----
`GARCHModel` scales returns by `scale` (default 100) before estimation for
numerical stability, so all reported coefficients refer to returns expressed in
**percent** — the conventional presentation in the GARCH literature. The slope
coefficients (alpha, beta, gamma, d, phi, delta) are scale-free; only the
intercepts (mu, omega) depend on the scaling, and they are reported on the
percent scale. `scale_note()` returns a ready-made caption stating this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    """Conventional significance markers."""
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------

def param_table(models: dict, *, include_stars: bool = True) -> pd.DataFrame:
    """
    Tidy coefficient table across fitted models.

    Parameters
    ----------
    models : dict mapping display name -> fitted GARCHModel
    include_stars : append a `signif` column with ***/**/* markers
                    (p < 0.01 / 0.05 / 0.10)

    Returns
    -------
    DataFrame with columns: model, parameter, coef, std_err, t_stat, p_value
    (and `signif`), one row per estimated coefficient. Models that are not
    fitted are skipped.
    """
    rows: list[dict] = []
    for name, m in models.items():
        res = getattr(m, "_result", None)
        if res is None:
            continue
        for param in res.params.index:
            p = float(res.pvalues[param])
            row = {
                "model": name,
                "parameter": param,
                "coef": float(res.params[param]),
                "std_err": float(res.std_err[param]),
                "t_stat": float(res.tvalues[param]),
                "p_value": p,
            }
            if include_stars:
                row["signif"] = _stars(p)
            rows.append(row)

    cols = ["model", "parameter", "coef", "std_err", "t_stat", "p_value"]
    if include_stars:
        cols.append("signif")
    return pd.DataFrame(rows, columns=cols)


# Conventional equation order for parameter columns, rather than alphabetical.
_PARAM_ORDER = ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]",
                "phi", "d", "beta", "delta", "nu", "lambda"]


def _order_and_reindex(wide: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Column order per `_PARAM_ORDER` (unknowns last), rows in `models` order."""
    ordered = [c for c in _PARAM_ORDER if c in wide.columns]
    ordered += [c for c in wide.columns if c not in ordered]
    wide = wide[ordered]
    return wide.reindex([n for n in models if n in wide.index])


def param_matrix(
    models: dict,
    *,
    decimals: int = 4,
    show: str = "std_err",
) -> pd.DataFrame:
    """
    Wide publication-style coefficient table: one row per model, one column per
    parameter, cells formatted as "coef (std_err)" with significance stars.

    Parameters
    ----------
    models   : dict of name -> fitted GARCHModel
    decimals : rounding for both coefficient and the bracketed quantity
    show     : what to put in brackets — "std_err", "t_stat" or "p_value"

    Empty cells mean the parameter does not exist for that specification (e.g.
    `gamma[1]` is absent from a symmetric GARCH, `nu` from a Normal model).
    """
    if show not in ("std_err", "t_stat", "p_value"):
        raise ValueError("show must be 'std_err', 't_stat' or 'p_value'")

    long = param_table(models, include_stars=True)
    if long.empty:
        return pd.DataFrame()

    fmt = f"{{:.{decimals}f}}"
    long["cell"] = (
        long["coef"].map(fmt.format)
        + long["signif"]
        + " (" + long[show].map(fmt.format) + ")"
    )
    wide = long.pivot(index="model", columns="parameter", values="cell")
    return _order_and_reindex(wide, models).fillna("")


def pvalue_matrix(models: dict, *, decimals: int | None = 4) -> pd.DataFrame:
    """
    Wide table of exact p-values: one row per model, one column per parameter.

    `param_matrix`'s stars bucket significance into three tiers, which hides
    exactly the cases worth citing precisely — a borderline p=0.041 and a
    solid p=0.0009 both show "**"/"***"; a near-miss p=0.099 and an
    unremarkable p=0.4 both show nothing. This gives the exact number instead,
    for citing precise values (e.g. in a referee response) or checking
    borderline cases directly.

    Empty (NaN) cells mean the parameter does not exist for that specification.
    Pass `decimals=None` to keep full float precision instead of rounding.
    """
    long = param_table(models, include_stars=False)
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot(index="model", columns="parameter", values="p_value")
    wide = _order_and_reindex(wide, models)
    return wide.round(decimals) if decimals is not None else wide


# ---------------------------------------------------------------------------
# Asymmetry (referee: "is the asymmetry parameter redundant?")
# ---------------------------------------------------------------------------

# Parameter carrying the leverage/asymmetry effect, per specification.
_ASYM_PARAM = "gamma[1]"


def asymmetry_summary(models: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Isolate the asymmetry coefficient of each model and report whether it is
    statistically distinguishable from zero.

    This answers the referee's question directly: if the leverage coefficient is
    insignificant, the asymmetric specification collapses (in practice) to its
    symmetric counterpart, which explains why a plain GARCH can forecast at
    least as well despite the asymmetric model's better in-sample fit.

    A model with no asymmetry parameter is reported with `has_asymmetry=False`
    — note that a specification which is *nominally* asymmetric but estimated
    with the asymmetric order set to zero will show up this way, which is
    itself worth catching.

    Returns
    -------
    DataFrame indexed by model: has_asymmetry, coef, std_err, t_stat, p_value,
    significant, verdict.
    """
    rows: dict[str, dict] = {}
    for name, m in models.items():
        res = getattr(m, "_result", None)
        if res is None:
            continue
        if _ASYM_PARAM in res.params.index:
            p = float(res.pvalues[_ASYM_PARAM])
            sig = bool(p < alpha)
            rows[name] = {
                "has_asymmetry": True,
                "coef": float(res.params[_ASYM_PARAM]),
                "std_err": float(res.std_err[_ASYM_PARAM]),
                "t_stat": float(res.tvalues[_ASYM_PARAM]),
                "p_value": p,
                "significant": sig,
                "verdict": ("leverage effect supported" if sig
                            else "not distinguishable from 0 -> effectively symmetric"),
            }
        else:
            rows[name] = {
                "has_asymmetry": False,
                "coef": np.nan, "std_err": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "significant": False,
                "verdict": "no asymmetry parameter estimated",
            }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persistence(models: dict) -> pd.DataFrame:
    """
    Volatility persistence per model, using the measure appropriate to each
    specification. Values are only comparable within the same measure.

    - GARCH        : alpha + beta
    - GJR-GARCH    : alpha + beta + gamma/2  (valid for symmetric innovations,
                     where the negative-shock indicator fires half the time)
    - EGARCH       : beta  (AR coefficient of the log-variance)
    - FIGARCH      : reports the long-memory parameter d instead; persistence is
                     hyperbolic, not geometric, so alpha+beta does not apply
    - APARCH       : NaN — persistence has no simple closed form (it depends on
                     delta and the innovation distribution)

    Returns
    -------
    DataFrame indexed by model: measure, value, (and `d` where applicable).
    """
    rows: dict[str, dict] = {}
    for name, m in models.items():
        res = getattr(m, "_result", None)
        if res is None:
            continue
        p = res.params
        mt = getattr(m, "model_type", "")

        if mt == "GARCH":
            val, measure = float(p.get("alpha[1]", np.nan)) + float(p.get("beta[1]", np.nan)), "alpha+beta"
        elif mt == "GJR-GARCH":
            val = (float(p.get("alpha[1]", np.nan))
                   + float(p.get("beta[1]", np.nan))
                   + 0.5 * float(p.get("gamma[1]", 0.0)))
            measure = "alpha+beta+gamma/2"
        elif mt == "EGARCH":
            val, measure = float(p.get("beta[1]", np.nan)), "beta (log-variance AR)"
        elif mt == "FIGARCH":
            val, measure = float(p.get("d", np.nan)), "d (long memory)"
        else:  # APARCH and anything else
            val, measure = np.nan, "n/a (no closed form)"

        rows[name] = {"measure": measure, "value": val}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Rendered equations
# ---------------------------------------------------------------------------

_DIST_LABEL = {"normal": r"\mathcal{N}(0,1)", "t": r"t_{\nu}", "ged": r"\mathrm{GED}(\nu)"}


def _f(v: float, decimals: int = 4) -> str:
    """Format a coefficient, using a leading sign for readability in equations."""
    return f"{v:.{decimals}f}"


def _signed(v: float, decimals: int = 4) -> str:
    return f"{'-' if v < 0 else '+'} {abs(v):.{decimals}f}"


def equation_lines(name: str, model, decimals: int = 4) -> list[str]:
    """
    LaTeX lines for one fitted model: the mean equation and the variance
    equation, with estimated coefficients substituted.

    The variance equation reflects the specification **as actually estimated**:
    the asymmetry term appears only if an asymmetry coefficient was estimated.
    """
    res = getattr(model, "_result", None)
    if res is None:
        raise RuntimeError(f"{name}: model is not fitted")
    p = res.params
    mt = getattr(model, "model_type", "")
    dist = getattr(model, "dist", "normal")
    z = _DIST_LABEL.get(dist, dist)

    mu = float(p.get("mu", np.nan))
    lines = [
        rf"\textbf{{{name}}}",
        rf"r_t = {_f(mu, decimals)} + \varepsilon_t, \qquad "
        rf"\varepsilon_t = \sigma_t z_t, \qquad z_t \sim {z}",
    ]

    om = float(p.get("omega", np.nan))
    al = float(p.get("alpha[1]", np.nan))
    be = float(p.get("beta[1]", p.get("beta", np.nan)))
    ga = float(p["gamma[1]"]) if "gamma[1]" in p.index else None

    if mt in ("GARCH", "GJR-GARCH"):
        eq = rf"\sigma_t^2 = {_f(om, decimals)} + {_f(al, decimals)}\,\varepsilon_{{t-1}}^2"
        if ga is not None:
            eq += (rf" {_signed(ga, decimals)}\,\varepsilon_{{t-1}}^2"
                   rf"\,\mathbb{{1}}_{{\{{\varepsilon_{{t-1}} < 0\}}}}")
        eq += rf" + {_f(be, decimals)}\,\sigma_{{t-1}}^2"
    elif mt == "EGARCH":
        eq = (rf"\ln \sigma_t^2 = {_f(om, decimals)} + {_f(al, decimals)}"
              rf"\left(|z_{{t-1}}| - \mathbb{{E}}|z_{{t-1}}|\right)")
        if ga is not None:
            eq += rf" {_signed(ga, decimals)}\,z_{{t-1}}"
        eq += rf" + {_f(be, decimals)}\,\ln \sigma_{{t-1}}^2"
    elif mt == "APARCH":
        de = float(p.get("delta", np.nan))
        inner = rf"|\varepsilon_{{t-1}}|"
        if ga is not None:
            inner = rf"\left(|\varepsilon_{{t-1}}| {_signed(-ga, decimals)}\,\varepsilon_{{t-1}}\right)"
        eq = (rf"\sigma_t^{{{_f(de, decimals)}}} = {_f(om, decimals)} + "
              rf"{_f(al, decimals)}\,{inner}^{{{_f(de, decimals)}}} + "
              rf"{_f(be, decimals)}\,\sigma_{{t-1}}^{{{_f(de, decimals)}}}")
    elif mt == "FIGARCH":
        phi = float(p.get("phi", np.nan))
        d = float(p.get("d", np.nan))
        eq = (rf"\sigma_t^2 = {_f(om, decimals)} + {_f(be, decimals)}\,\sigma_{{t-1}}^2 + "
              rf"\left[1 - {_f(be, decimals)}L - (1 - {_f(phi, decimals)}L)"
              rf"(1-L)^{{{_f(d, decimals)}}}\right]\varepsilon_t^2")
    else:
        eq = r"\text{(unrecognised specification)}"

    lines.append(eq)

    if "nu" in p.index:
        lines.append(rf"\nu = {_f(float(p['nu']), decimals)}")
    return lines


def equations_markdown(models: dict, decimals: int = 4) -> str:
    """
    Markdown/LaTeX block with the fitted equations of every model, ready for
    `display(Markdown(...))` in a notebook.
    """
    blocks: list[str] = []
    for name, m in models.items():
        if getattr(m, "_result", None) is None:
            continue
        lines = equation_lines(name, m, decimals=decimals)
        body = r" \\[2pt] ".join(lines)
        blocks.append(rf"$$\begin{{aligned}} {body} \end{{aligned}}$$")
    return "\n\n".join(blocks)


def scale_note(models: dict) -> str:
    """
    Caption stating the units the coefficients refer to, derived from the
    models' `scale` attribute (they should all share one).
    """
    scales = {getattr(m, "scale", None) for m in models.values()}
    scales.discard(None)
    if scales == {100.0}:
        return ("Coefficients are estimated on returns expressed in percent "
                "(log-returns x 100), the conventional scaling in the GARCH "
                "literature. Slope coefficients (alpha, beta, gamma, d, phi, "
                "delta) are scale-free; the intercepts (mu, omega) are on the "
                "percent scale. Standard errors in brackets; "
                "*** p<0.01, ** p<0.05, * p<0.10.")
    return (f"Coefficients estimated on returns scaled by {sorted(scales)}. "
            "Standard errors in brackets; *** p<0.01, ** p<0.05, * p<0.10.")
