"""
Pinned copies of the Yahoo Finance download.

WHY
---
Yahoo revises its history: a re-download months later can return different
prices for the same dates, and then the article's tables cannot be rebuilt
from the code alone. A snapshot freezes the OHLC series the first time it is
downloaded, and every later run reads that file instead of the network.

MODES (the notebook's DATA_SNAPSHOT)
------------------------------------
  "auto"     load the snapshot if it exists; otherwise download and save it
  "refresh"  download again and overwrite the snapshot
  "off"      always download, never read or write a snapshot (the behaviour
             before October 2026)

FILES
-----
One CSV per ticker and date range, `<TICKER>_<start>_<end>.csv`, next to a
`.json` sidecar with the row count, the date span, the download time, the
yfinance version and the SHA-256 of the CSV. The hash is checked on every load,
so a snapshot that was edited by hand fails loudly instead of feeding the
models silently changed data.

Relative directories are resolved against the REPOSITORY ROOT, not the folder
the notebook happens to run in, so a notebook copied into experiments/<date>/
reads the same snapshot as the one in the root. The default directory is
<repo>/data.

Floats are written with full precision and read back with
float_precision="round_trip", so a snapshot reproduces the download bit for
bit (pinned by the tests).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

SNAPSHOT_MODES = ("auto", "refresh", "off")
OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def repo_root() -> Path:
    """The repository root: the folder that contains the volatility_pipeline package."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path) -> Path:
    """Absolute paths unchanged; relative ones taken from the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root() / p


def snapshot_path(ticker: str, start: str, end: str, directory=None) -> Path:
    base = resolve_repo_path(directory if directory is not None else "data")
    return base / f"{ticker}_{start}_{end}.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yahoo_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """The download the notebooks always made, reduced to the four OHLC columns."""
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {ticker} {start}..{end}. Check the "
            f"connection, or use an existing snapshot (DATA_SNAPSHOT='auto')."
        )
    df = pd.DataFrame({c: raw[c].squeeze() for c in OHLC_COLUMNS})
    df.index.name = "Date"
    return df


def _write(df: pd.DataFrame, path: Path, meta: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    meta = {**meta, "sha256": _sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
    df.index.name = "Date"
    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"snapshot {path} lacks columns {missing}.")
    return df[list(OHLC_COLUMNS)].astype(float)


def load_ohlc(
    ticker: str,
    start: str,
    end: str,
    *,
    snapshot: str = "auto",
    directory=None,
    downloader: Callable[[str, str, str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Daily OHLC for `ticker` from `start` (inclusive) to `end` (exclusive, as
    yfinance treats it), and a dict saying where it came from.

    Returns (DataFrame with columns Open, High, Low, Close on a DatetimeIndex,
    info) where info has "source" ("snapshot" or "yahoo"), "path", "sha256",
    "rows", "first", "last" and, for a snapshot, when it was downloaded.

    `downloader` replaces the Yahoo call (tests use it to stay offline).
    """
    if snapshot not in SNAPSHOT_MODES:
        raise ValueError(f"snapshot must be one of {SNAPSHOT_MODES}, got {snapshot!r}.")
    fetch = downloader or _yahoo_download
    path = snapshot_path(ticker, start, end, directory)

    if snapshot == "auto" and path.exists():
        side = path.with_suffix(".json")
        if not side.exists():
            raise ValueError(
                f"snapshot {path} has no {side.name} sidecar, so its integrity cannot be "
                f"checked. Restore the sidecar, or set DATA_SNAPSHOT='refresh' to re-download."
            )
        meta = json.loads(side.read_text())
        actual = _sha256(path)
        if actual != meta.get("sha256"):
            raise ValueError(
                f"snapshot {path} does not match the SHA-256 recorded when it was saved: the "
                f"file was changed after download. Restore it, or set "
                f"DATA_SNAPSHOT='refresh' to download a new one deliberately."
            )
        df = _read(path)
        return df, {**meta, "source": "snapshot", "path": str(path)}

    df = fetch(ticker, start, end)
    df = df[list(OHLC_COLUMNS)].astype(float)
    df.index.name = "Date"
    info = {
        "ticker": ticker, "start": start, "end": end, "rows": int(len(df)),
        "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance": _yfinance_version(),
    }
    if snapshot == "off":
        return df, {**info, "source": "yahoo", "path": None, "sha256": None}
    meta = _write(df, path, info)
    # Read back what was written, so a run that creates the snapshot sees
    # exactly the data every later run will load from it.
    return _read(path), {**meta, "source": "yahoo (snapshot saved)", "path": str(path)}


def _yfinance_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("yfinance")
    except Exception:
        return None


def describe_source(info: dict) -> str:
    """One line for the notebook's data cell."""
    where = info.get("path") or "not saved (DATA_SNAPSHOT='off')"
    when = f", downloaded {info['downloaded_utc']}" if info.get("downloaded_utc") else ""
    sha = f", sha256 {info['sha256'][:12]}…" if info.get("sha256") else ""
    return (f"Data source: {info['source']}  |  {info['rows']} rows {info['first']} -> {info['last']}"
            f"{when}{sha}\n  snapshot file: {where}")
