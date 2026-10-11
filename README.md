# volatility-pipeline

Out-of-sample volatility forecasting for energy futures (Brent BZ=F, Henry Hub
NG=F): the GARCH family, Realized GARCH, XGBoost and LSTM, and GARCH-family
combiners, evaluated on one rolling one-step-ahead protocol with DM and MCS
inference.

- `volatility_pipeline/` — the package: data preparation (contract rolls,
  proxies, snapshots), models, evaluation and tests of predictive accuracy.
- `Main_pipeline_xgb_lstm.ipynb` — the full pipeline. Its defaults are the
  article's frozen specification (October 2026); `main_pipeline.ipynb`
  (GARCH family), `main_pipeline_xgb.ipynb` and `lstm_pipeline.ipynb` are
  subsets.
- `experiments/README.md` — status of every experiment batch and how to run the
  final experiments.
- `CHANGELOG.md` — every change that alters results, and which earlier results
  it invalidates.

Environment: `/opt/anaconda3/envs/volatility-pipeline-new` (Python 3.14;
`requirements.txt`). Tests: `pytest tests/`.
