"""
Tests for the data snapshot and the results export.

The snapshot exists so a later Yahoo revision cannot move the article's tables;
the export so they can be rebuilt and checked without re-running the models.
The tests stay offline: a fake downloader stands in for Yahoo and counts calls.
"""
import json

import numpy as np
import pandas as pd
import pytest

from volatility_pipeline.data import load_ohlc, describe_source, repo_root, resolve_repo_path
from volatility_pipeline.evaluation import ForecastResult, collect_config, export_run


def _fake_ohlc(n=250, seed=0):
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date")
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


class FakeYahoo:
    def __init__(self, frame):
        self.frame, self.calls = frame, 0

    def __call__(self, ticker, start, end):
        self.calls += 1
        return self.frame.copy()


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def test_auto_downloads_once_then_reads_the_snapshot(tmp_path):
    yahoo = FakeYahoo(_fake_ohlc())
    df1, info1 = load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=yahoo)
    df2, info2 = load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=yahoo)
    assert yahoo.calls == 1
    assert info1["source"].startswith("yahoo") and info2["source"] == "snapshot"
    assert (tmp_path / "NG=F_2020-01-01_2021-01-01.csv").exists()
    assert (tmp_path / "NG=F_2020-01-01_2021-01-01.json").exists()
    pd.testing.assert_frame_equal(df1, df2)


def test_snapshot_reproduces_the_download_bit_for_bit(tmp_path):
    raw = _fake_ohlc(seed=3)
    df, _ = load_ohlc("BZ=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=FakeYahoo(raw))
    np.testing.assert_array_equal(df.to_numpy(), raw.to_numpy())
    assert df.index.equals(raw.index)
    again, _ = load_ohlc("BZ=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=FakeYahoo(raw))
    np.testing.assert_array_equal(again.to_numpy(), raw.to_numpy())


def test_refresh_overwrites_and_off_never_touches_disk(tmp_path):
    yahoo = FakeYahoo(_fake_ohlc(seed=1))
    load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=yahoo)
    yahoo.frame = _fake_ohlc(seed=2)                                     # Yahoo revised its data
    df, _ = load_ohlc("NG=F", "2020-01-01", "2021-01-01", snapshot="refresh",
                      directory=tmp_path, downloader=yahoo)
    np.testing.assert_array_equal(df.to_numpy(), _fake_ohlc(seed=2).to_numpy())
    assert yahoo.calls == 2
    off_dir = tmp_path / "off"
    df, info = load_ohlc("NG=F", "2020-01-01", "2021-01-01", snapshot="off",
                         directory=off_dir, downloader=yahoo)
    assert info["source"] == "yahoo" and info["path"] is None and not off_dir.exists()


def test_a_changed_date_range_gets_its_own_snapshot(tmp_path):
    yahoo = FakeYahoo(_fake_ohlc())
    load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=yahoo)
    load_ohlc("NG=F", "2020-01-01", "2021-06-30", directory=tmp_path, downloader=yahoo)
    assert yahoo.calls == 2 and len(list(tmp_path.glob("*.csv"))) == 2


def test_an_edited_snapshot_fails_loudly(tmp_path):
    load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=FakeYahoo(_fake_ohlc()))
    csv = tmp_path / "NG=F_2020-01-01_2021-01-01.csv"
    lines = csv.read_text().splitlines()
    lines[5] = lines[5].replace(lines[5].split(",")[-1], "999.0")
    csv.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=FakeYahoo(_fake_ohlc()))
    csv.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path, downloader=FakeYahoo(_fake_ohlc()))


def test_relative_directories_are_taken_from_the_repository_root():
    assert resolve_repo_path("data") == repo_root() / "data"
    assert (repo_root() / "volatility_pipeline" / "__init__.py").exists()


def test_bad_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="snapshot must be"):
        load_ohlc("NG=F", "2020-01-01", "2021-01-01", snapshot="on", directory=tmp_path)


def test_describe_source_names_the_file(tmp_path):
    _, info = load_ohlc("NG=F", "2020-01-01", "2021-01-01", directory=tmp_path,
                        downloader=FakeYahoo(_fake_ohlc()))
    assert str(tmp_path) in describe_source(info)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@pytest.fixture
def results():
    idx = pd.PeriodIndex(pd.bdate_range("2022-07-01", periods=40), freq="D")
    rng = np.random.default_rng(4)
    act = pd.Series(rng.lognormal(-8, 0.5, 40), index=idx)
    return {n: ForecastResult(n, pd.Series(act.mean() * s, index=idx), act, meta={"n_refits": 4})
            for n, s in [("GARCH-NORMAL", 1.0), ("XGB-COMB", 1.3), ("GARCH-EW", 0.9)]}


def test_export_writes_forecasts_losses_metrics_tables_and_config(tmp_path, results):
    tables = {"dm_vs_benchmark": pd.DataFrame({"p_value": [0.1]}, index=["XGB-COMB"]),
              "robustness": pd.DataFrame([[1.0, 2.0]], columns=pd.MultiIndex.from_tuples(
                  [("main", "QLIKE"), ("scored on r^2", "QLIKE")])),
              "skipped": None}
    out = export_run(tmp_path / "run", results, tables=tables, config={"TICKER": "NG=F"},
                     extra={"roll_days": ["2026-01-29"]}, group_of=lambda n: n.split("-")[0])
    files = {p.name for p in out.iterdir()}
    assert files == {"forecasts.csv", "losses_qlike.csv", "losses_squared.csv", "metrics.csv",
                     "dm_vs_benchmark.csv", "robustness.csv", "run_config.json"}
    fc = pd.read_csv(out / "forecasts.csv", index_col=0)
    np.testing.assert_allclose(fc["actual"], results["GARCH-NORMAL"].actuals.to_numpy())
    np.testing.assert_allclose(fc["XGB-COMB"], results["XGB-COMB"].forecasts.to_numpy())
    lq = pd.read_csv(out / "losses_qlike.csv", index_col=0)
    np.testing.assert_allclose(lq["GARCH-EW"], results["GARCH-EW"].loss_series("qlike").to_numpy())
    met = pd.read_csv(out / "metrics.csv", index_col=0)
    assert list(met.columns) == ["group", "n", "QLIKE", "RMSE", "rank_QLIKE"]
    assert np.isclose(met.loc["XGB-COMB", "QLIKE"], results["XGB-COMB"].metrics()["QLIKE"])
    assert list(pd.read_csv(out / "robustness.csv", index_col=0).columns) == \
        ["main | QLIKE", "scored on r^2 | QLIKE"]
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["config"] == {"TICKER": "NG=F"} and cfg["n_models"] == 3
    assert cfg["extra"]["roll_days"] == ["2026-01-29"]
    assert len(cfg["provenance"]["git_commit"]) == 40
    assert set(cfg["files"]) == files


def test_reexport_replaces_only_its_own_files(tmp_path, results):
    out = export_run(tmp_path, results, tables={"old_table": pd.DataFrame({"a": [1]})})
    (out / "notes.txt").write_text("mine")
    export_run(tmp_path, results)                         # second run without old_table
    files = {p.name for p in out.iterdir()}
    assert "old_table.csv" not in files and "notes.txt" in files


def test_export_refuses_unaligned_results(tmp_path, results):
    r = results["XGB-COMB"]
    results["XGB-COMB"] = ForecastResult("XGB-COMB", r.forecasts.iloc[1:], r.actuals.iloc[1:])
    with pytest.raises(ValueError, match="different dates"):
        export_run(tmp_path, results)
    with pytest.raises(ValueError, match="empty"):
        export_run(tmp_path, {})


def test_collect_config_keeps_plain_upper_case_settings():
    ns = {"TICKER": "NG=F", "GARCH_SPECS": [("GARCH", "normal")], "N_BOOT": np.int64(1000),
          "lower": 1, "_PRIVATE": 2, "PD": pd, "FUNC": len, "TABLE": pd.DataFrame(), "W": None}
    cfg = collect_config(ns)
    assert cfg == {"GARCH_SPECS": [["GARCH", "normal"]], "N_BOOT": 1000, "TICKER": "NG=F", "W": None}
    json.dumps(cfg)
