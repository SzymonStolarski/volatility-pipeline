# Changelog

Changes that alter results, newest first. Each entry names the pull request or
commit, what changed, and why. The last section states which experiment folders
these changes invalidate or supersede; `experiments/README.md` keeps the status
of every folder.

## October 2026 — the article's frozen specification (PRs #16–#20)

- **#20 Final-run setup.** Data snapshot (`DATA_SNAPSHOT`): the Yahoo download
  is pinned in `<repo>/data` with a SHA-256 checked on every load. Results
  export (`EXPORT_RESULTS`): forecasts, per-day losses, metrics and every table,
  plus `run_config.json` with all settings, the data hash, the git commit and
  package versions. The notebook defaults are now the frozen specification;
  every option it switches off is kept as a commented-out line or listed next
  to its setting.
- **#19 GARCH-family combiner.** `XGB-COMB` and `LSTM-COMB` receive the
  one-step forecasts of all GARCH specifications (training on in-sample
  conditional variances, forecasting on genuine one-step forecasts), with the
  equal-weight mean `GARCH-EW` as the benchmark. Single-base hybrids are
  unchanged: bit-identical forecasts with one specification.
- **#18 The article's tests.**
  - DM only against GARCH(1,1)-N.
  - The MCS block length is stated and shared by both implementations. The
    own MCS used T^(1/3) = 10 while arch used sqrt(T) = 31; both now use
    sqrt(T).
  - `N_BOOT` 2000 → 1000.
  - The PIT is decided by Anderson–Darling with a Monte Carlo p-value; KS is
    supplementary.
  - One tested chronological split for every tuning hold-out.
  - The tuned hyperparameters are recorded on the results.
- **#17 Fixed lag window.** XGBoost and the LSTM both see 10 lagged returns and
  10 lagged squared returns, never tuned. The LSTM lookback had been an Optuna
  dimension over {5, 10, 20, 40}. The committed defaults had been
  `XGB_N_LAGS = 5`, `HYBRID_N_LAGS = 5`, `LSTM_LOOKBACK = 20`.
- **#16 Contract rolls and the proxy floor. Changes every number.**
  - Roll days come from the exchange calendar (NYMEX NG: the day after
    expiry; NYMEX BZ: the first trading day of each month).
  - On a roll day the return is ln(C/O) for every model, and the overnight
    term of the proxy and of the realized measure is 0. Robustness:
    `ROLL_HANDLING = "drop"`.
  - One floor: the 1% quantile of positive training-window values.
  - On NG=F the 13.09 lead of Realized GARCH over the best plain GARCH falls
    from 0.096 to 0.015 QLIKE (DM p 0.006 → 0.088).

## September 2026 — the supervisor's September review (PRs #11–#15)

- **#15 Symmetric tuning.** XGBoost had been tuned at every re-estimation
  (about 101 searches per model), the LSTM never. Both are now tuned once, at
  the first estimation (`*_TUNE = "first"`).
- **#14 Realized GARCH** (Hansen, Huang & Shek 2012), written from scratch,
  with a selectable realized measure.
- **#13 ML training target** decoupled from the scoring proxy
  (`ML_TARGET_PROXY`).
- **#12 Overnight-inclusive proxies** (GK+overnight, RS+overnight), the proxy
  registry and a contract-roll screen.
- **#11 Residual-level diagnostics.** Normality and PIT tests run on the
  standardized residuals, where the distributional assumption lives.

## August 2026 — evaluator fixes (PR #10)

- **fb82dc1 (9 Aug) — stale forecasts.** `RollingEvaluator` did not advance any
  model's state between re-estimations. Within each 10-day block every model
  repeated the forecast made at the block origin, so the "one-step" evaluation
  was a mixture of 1- to 10-step forecasts. Every model now implements
  `update()`, and the evaluator refuses models that do not. On identical
  configurations (Brent, expanding, Garman–Klass), 15/15 GARCH models improved
  on both QLIKE and RMSE, and the post-fix Brent RMSE reproduces the first
  article's Table 3 to within 0.62%.
- **fb82dc1, c4e6b8f — ML target.** The ML models are trained on the scoring
  proxy instead of squared returns.

## July 2026

- **565221a (30 Jul, PR #9) — LSTM retransformation.** A network trained on a
  log-variance target estimates the conditional geometric mean, about 0.2 times
  the variance level. A Duan (1983) smearing factor now corrects it.
- **489578d (26 Jul, PR #8) — asymmetric EGARCH and APARCH.** Both had been
  estimated WITHOUT their leverage term (arch's default `o = 0`), despite being
  labelled asymmetric. They now use `o = 1` (`ASYM_ORDER`).

## Invalidated and superseded results

The following experiment folders are formally withdrawn. Numbers from them must
not be quoted.

| Folder | Status | Reason |
|---|---|---|
| `experiments/6.06.2026` | **void** | stale forecasts (before fb82dc1); EGARCH/APARCH without leverage (before 489578d) |
| `experiments/7.07.2026` | **void** | as 6.06: before fb82dc1 and before 489578d |
| `experiments/1.08.2026` | **void** | stale forecasts (before fb82dc1) |
| `experiments/9.08.2026` | superseded | intraday-only Garman–Klass proxy; XGBoost tuned at every re-estimation while the LSTM was never tuned; contract rolls untreated |
| `experiments/13.09.2026` | superseded | contract rolls untreated in returns and proxy; no proxy floor; LSTM lookback tuned; single-base hybrids; 15 specifications with GED |
