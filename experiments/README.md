# Experiments

One folder per batch of runs, named by date. The notebooks in a folder are copies
of the repository notebooks with the configuration used for that batch.

## Status

| Folder | Status | Notes |
|---|---|---|
| `6.06.2026` | void | before the stale-forecast fix (fb82dc1) and before EGARCH/APARCH got their leverage term (489578d) |
| `7.07.2026` | void | before fb82dc1 and 489578d |
| `1.08.2026` | void | before fb82dc1 (forecasts stale within each re-estimation block) |
| `9.08.2026` | superseded | intraday-only Garman–Klass proxy, asymmetric tuning, contract rolls untreated |
| `13.09.2026` | superseded | contract rolls untreated, no proxy floor, tuned LSTM lookback, single-base hybrids, GED |

Details and commits: `../CHANGELOG.md`.

## Running the final experiments

The defaults of `Main_pipeline_xgb_lstm.ipynb` are the article's frozen
specification (October 2026), so a final run only needs the commodity and the
output folder set.

1. **Code.** Work from `main` at the commit to be reported; `pytest tests/`
   must pass. Every export records the commit and whether the tree had
   uncommitted changes, so commit before running.
2. **Notebooks.** Copy `Main_pipeline_xgb_lstm.ipynb` into
   `experiments/<date>/` twice, one copy per commodity, and in each copy set:
   ```python
   TICKER      = "BZ=F"                          # and "NG=F" in the other copy
   EXPORT_ROOT = "experiments/<date>/results"
   ```
   Leave everything else as it is, including `RUN_LABEL = "main"`. Each run
   writes into its own folder, `experiments/<date>/results/<TICKER>/<YYYY-MM-DD_HHMM>_main/`,
   timestamped when the config cell runs. Run the copies with the repository
   root as the working directory, as for the earlier batches, since the
   `volatility_pipeline` import needs it. Snapshot and export paths are
   resolved against the repository root either way.
3. **Data.** `DATA_SNAPSHOT = "auto"` reads the pinned download in `<repo>/data`.
   The snapshots were created on 11 October 2026 and reproduce the
   series behind every check made in October:

   | File | Rows | Span | SHA-256 |
   |---|---|---|---|
   | `BZ=F_2010-01-01_2026-06-30.csv` | 4115 | 2010-01-04 – 2026-06-29 | `252e2a50db54168e1a1cfd9d2439e79dad16e4e77ba37805cdce78a20797d045` |
   | `NG=F_2010-01-01_2026-06-30.csv` | 4147 | 2010-01-04 – 2026-06-29 | `5e0a1f6b594383da1fc19d77ee379d023d0382a37a234bfa2ad62f0380fc2434` |

   `<repo>/data` is git-ignored, since Yahoo Finance data is not ours to
   redistribute. Keep a private copy, or force-add the files (`git add -f`) if
   the repository is private.
4. **Robustness runs for the supplement.** Each is one more copy, with one
   setting changed and its own label:
   - contract-roll days removed from the data: `ROLL_HANDLING = "drop"`,
     `RUN_LABEL = "roll-drop"`;
   - sliding estimation window: `WINDOW_TYPE = "sliding"`, `WINDOW_SIZE = 750`,
     `RUN_LABEL = "sliding750"`.

   They land next to the main runs, e.g. `results/NG=F/2026-10-20_1340_roll-drop/`,
   and cannot overwrite them. Scoring on r² and scoring without roll days need
   no separate run. Every run reports them in its robustness table.
5. **Commit** the notebooks together with their `results/` folders. Each run
   folder contains `run_config.json` with all settings, the data hash, the git
   commit and the package versions. If a run was repeated, keep only the
   folder of the run that is reported, or note here why both are kept.

## The frozen specification and where it is set

| Item | Specification | Setting |
|---|---|---|
| Data | BZ=F and NG=F, daily; train 2010-01 – 2022-06, test 2022-07 – 2026-06 (1004 days) | `TICKER`, `FULL_DATA_*`, `TRAIN_END`, `TEST_START` |
| Estimation | expanding window, re-estimation every 10 days | `WINDOW_TYPE = "expanding"`, `REFIT_EVERY = 10` |
| Contract rolls | open-to-close return on calendar roll days, overnight term 0 | `ROLL_HANDLING = "adjust"` |
| Proxy | Garman–Klass + squared overnight return, floored at the 1% quantile of positive training values | `VARIANCE_PROXY = "garman_klass_overnight"`, `PROXY_FLOOR_Q = 0.01` |
| Losses | QLIKE (main), RMSE | `TEST_LOSS = "qlike"`, `REPORT_METRICS = ["QLIKE", "RMSE"]` |
| GARCH family | GARCH, GJR, EGARCH, APARCH, FIGARCH, all (1,1), Normal and t | `GARCH_SPECS` (GED lines commented out), `ASYM_ORDER = 1` |
| Realized GARCH | one variant, GK+overnight measure | `REALIZED_GARCH_SPECS = ["garman_klass_overnight"]` |
| ML | XGBoost, LSTM; 10 lagged returns and 10 lagged squared returns; target log(GK+overnight), floored, Duan correction; tuned once (50 / 20 trials) | `ML_N_LAGS = 10`, `ML_TARGET_PROXY = None`, `*_TUNE = "first"`, `XGB_N_TRIALS = 50`, `LSTM_N_TRIALS = 20` |
| Hybrids | nonlinear combiners of the 10 GARCH forecasts; no single-base hybrids | `COMBINER_MODELS = ["xgb", "lstm"]`, `HYBRID_SPECS = []` |
| Benchmark | equal-weight mean of the 10 GARCH forecasts | `GARCH_EW = True` |
| Tests | MCS T_max, stationary bootstrap, 1000 replications, α = 0.05; DM (HLN) against GARCH(1,1)-N | `MCS_ALPHA`, `N_BOOT = 1000`, `MCS_BLOCK_SIZE = "sqrt"`, `DM_BENCHMARK = "GARCH-NORMAL"` |
| Models per commodity | 10 GARCH + Realized GARCH + GARCH-EW + XGB + LSTM + 2 combiners = 16 | — |
