"""
Write one run's results to disk, so the article's tables can be rebuilt (and
checked) without re-running the models.

`export_run` writes into one folder:

  forecasts.csv        one row per test date: the scoring proxy ("actual") and
                       every model's one-step variance forecast
  losses_<loss>.csv    the per-day loss of every model (QLIKE and squared error
                       by default): everything DM and the MCS are computed from
  metrics.csv          per model: group, number of forecasts, the reported
                       metrics and the QLIKE rank
  <table>.csv          any further tables the notebook passes (coefficients,
                       residual diagnostics, DM, MCS, robustness, ...)
  run_config.json      every notebook setting, the data source and its SHA-256,
                       the git commit (and whether the tree had uncommitted
                       changes), package versions, and the list of files written

`run_directory` names the folder: one per run,
<root>/<ticker>/<YYYY-MM-DD_HHMM>_<label> (e.g. results/NG=F/2026-10-20_1432_main),
so a later run (a robustness run, a repeat) never overwrites an earlier one.

Relative directories are resolved against the repository root. Re-exporting
into the same folder first removes the files the previous export listed in its
run_config.json, so the folder never mixes two runs; files the export did not
write are left alone.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.snapshot import repo_root, resolve_repo_path

_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "arch", "xgboost", "torch",
             "optuna", "scikit-learn", "yfinance")


def _jsonable(v):
    """JSON-safe copy of a config value, or a sentinel when it is not data."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        items = [_jsonable(x) for x in v]
        return _SKIP if any(x is _SKIP for x in items) else items
    if isinstance(v, dict):
        items = {str(k): _jsonable(x) for k, x in v.items()}
        return _SKIP if any(x is _SKIP for x in items.values()) else items
    return _SKIP


_SKIP = object()
_UNSAFE_IN_FOLDER_NAME = re.compile(r"[^\w.-]+")


def run_directory(root, ticker: str, label: str, stamp: str | None = None) -> Path:
    """
    The folder of one run: <root>/<ticker>/<stamp>_<label>, e.g.
    results/NG=F/2026-10-20_1432_main.

    `stamp` defaults to the current local time, YYYY-MM-DD_HHMM, so every run
    gets its own folder. Call this once, in the notebook's config cell: the
    whole notebook re-run makes a new folder, while re-running only the export
    cell writes into the same one. `label` says what the run is ("main",
    "roll-drop", "sliding750", ...); characters that do not belong in a folder
    name become "-". A folder that already exists belongs to an earlier run
    (this one has not exported yet), so a run repeated within the same minute
    gets "-2", "-3", ... instead of replacing it. The path is returned as given
    (a relative root stays relative); export_run resolves it against the
    repository root.
    """
    name = _UNSAFE_IN_FOLDER_NAME.sub("-", str(label)).strip("-.")
    if not name:
        raise ValueError(f"RUN_LABEL {label!r} leaves nothing usable as a folder name.")
    if stamp is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = Path(root) / ticker / f"{stamp}_{name}"
    candidate, k = folder, 2
    while resolve_repo_path(candidate).exists():
        candidate = folder.with_name(f"{folder.name}-{k}")
        k += 1
    return candidate


def collect_config(namespace: dict) -> dict:
    """
    The run's settings: every UPPER_CASE name in `namespace` (pass the
    notebook's globals()) whose value is plain data. Modules, functions,
    DataFrames and the like are skipped.
    """
    out = {}
    for k in sorted(namespace):
        if not k.isupper() or k.startswith("_"):
            continue
        v = _jsonable(namespace[k])
        if v is not _SKIP:
            out[k] = v
    return out


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def run_provenance() -> dict:
    """When, where and with which code and packages the run was made."""
    from importlib.metadata import version, PackageNotFoundError
    pkgs = {}
    for p in _PACKAGES:
        try:
            pkgs[p] = version(p)
        except PackageNotFoundError:
            pkgs[p] = None
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_uncommitted_changes": (bool(status) if status is not None else None),
        "packages": pkgs,
    }


def _plain_frame(table) -> pd.DataFrame:
    """A DataFrame from a DataFrame, a pandas Styler or a Series; flat columns."""
    if hasattr(table, "data") and isinstance(getattr(table, "data"), pd.DataFrame):
        table = table.data                                   # pandas Styler
    if isinstance(table, pd.Series):
        table = table.to_frame()
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"cannot export a {type(table).__name__} as a table.")
    df = table.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(x) for x in col if str(x)) for col in df.columns]
    return df


def export_run(
    directory,
    results: dict,
    *,
    tables: dict | None = None,
    config: dict | None = None,
    extra: dict | None = None,
    group_of=None,
    metrics=("QLIKE", "RMSE"),
    losses=("qlike", "squared"),
) -> Path:
    """
    Write `results` (name -> ForecastResult) and the given `tables`
    (name -> DataFrame / Styler) into `directory`; return the folder.

    `group_of(name) -> str` adds a group column to metrics.csv (pass the
    notebook's _group). `metrics` picks the columns of metrics.csv (the
    specification reports QLIKE and RMSE). `extra` lands in run_config.json
    as is (data source, roll days, ...).
    """
    if not results:
        raise ValueError("nothing to export: `results` is empty.")
    out = resolve_repo_path(directory)
    out.mkdir(parents=True, exist_ok=True)

    manifest = out / "run_config.json"
    if manifest.exists():
        try:
            for name in json.loads(manifest.read_text()).get("files", []):
                f = out / name
                if f.is_file() and f.parent == out:
                    f.unlink()
        except (ValueError, OSError):
            pass

    names = list(results)
    first = results[names[0]]
    for n in names[1:]:
        if not results[n].forecasts.index.equals(first.forecasts.index):
            raise ValueError(f"{n!r} was evaluated on different dates than {names[0]!r}.")
    written = []

    fc = pd.DataFrame({"actual": first.actuals, **{n: results[n].forecasts for n in names}})
    fc.index.name = "date"
    fc.to_csv(out / "forecasts.csv")
    written.append("forecasts.csv")

    for loss in losses:
        lf = pd.DataFrame({n: results[n].loss_series(loss) for n in names})
        lf.index.name = "date"
        lf.to_csv(out / f"losses_{loss}.csv")
        written.append(f"losses_{loss}.csv")

    rows = []
    for n in names:
        m = results[n].metrics()
        row = {"model": n}
        if group_of is not None:
            row["group"] = group_of(n)
        row["n"] = len(results[n].forecasts)
        row.update({k: m[k] for k in metrics})
        rows.append(row)
    met = pd.DataFrame(rows).set_index("model")
    if "QLIKE" in met.columns:
        met["rank_QLIKE"] = met["QLIKE"].rank(method="min").astype(int)
        met = met.sort_values("QLIKE")
    met.to_csv(out / "metrics.csv")
    written.append("metrics.csv")

    for tname, table in (tables or {}).items():
        if table is None:
            continue
        fname = f"{tname}.csv"
        _plain_frame(table).to_csv(out / fname)
        written.append(fname)

    meta = {
        "config": config or {},
        "provenance": run_provenance(),
        "proxy": first.proxy,
        "n_models": len(names),
        "models": names,
        "test_period": [str(first.forecasts.index[0]), str(first.forecasts.index[-1])],
        "n_test": len(first.forecasts),
        "extra": extra or {},
        "files": written + ["run_config.json"],
    }
    manifest.write_text(json.dumps(meta, indent=2, default=str))
    return out
