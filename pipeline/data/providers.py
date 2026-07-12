"""OHLCV data providers.

Every provider returns the same normalised frame:
    DatetimeIndex (UTC, ascending, unique) named 'dt'
    columns: open, high, low, close, volume  (float64)

Providers (chosen for keyless availability from this environment):
    coinbase  - Coinbase Exchange public candles (crypto; paginated, multi-year)
    bitstamp  - Bitstamp public OHLC (crypto; paginated)
    kraken    - Kraken public OHLC (crypto; last ~720 bars per interval)
    histdata  - histdata.com free forex/metals M1 archives (XAUUSD, EURUSD, ...)
    csv       - local CSV, including MetaTrader 5 "Save as" exports

Downloads are cached under data_cache/ so window permutations re-slice the
same immutable download instead of hitting the network repeatedly.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[2] / "data_cache"

COLS = ["open", "high", "low", "close", "volume"]

def _ts(dt) -> pd.Timestamp:
    """tz-aware UTC Timestamp from datetime that may or may not carry tzinfo."""
    t = pd.Timestamp(dt)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "dt"
    # drop non-positive prices and bars violating high>=low
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df.high >= df.low)
    return df[ok]


# ---------------------------------------------------------------- coinbase
def _coinbase(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    gran = _GRANULARITY[timeframe]
    out, t0 = [], start
    step = timedelta(seconds=gran * 300)  # API max 300 candles/request
    while t0 < end:
        t1 = min(t0 + step, end)
        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{symbol}/candles",
            params={"granularity": gran, "start": t0.isoformat(), "end": t1.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
            out.append(df)
        t0 = t1
        time.sleep(0.15)  # public rate limit ~10 req/s; stay well under
    if not out:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(out)
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    return _normalise(df)


# ---------------------------------------------------------------- bitstamp
def _bitstamp(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    gran = _GRANULARITY[timeframe]
    pair = symbol.replace("-", "").replace("/", "").lower()
    out, t0 = [], int(start.timestamp())
    t_end = int(end.timestamp())
    while t0 < t_end:
        r = requests.get(
            f"https://www.bitstamp.net/api/v2/ohlc/{pair}/",
            params={"step": gran, "limit": 1000, "start": t0},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("data", {}).get("ohlc", [])
        if not rows:
            break
        df = pd.DataFrame(rows)
        out.append(df)
        last = int(df["timestamp"].astype(int).max())
        if last <= t0:
            break
        t0 = last + gran
        time.sleep(0.2)
    if not out:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(out)
    df.index = pd.to_datetime(df.pop("timestamp").astype(int), unit="s", utc=True)
    df = df[df.index <= _ts(end)]
    return _normalise(df)


# ---------------------------------------------------------------- kraken
def _kraken(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    minutes = _GRANULARITY[timeframe] // 60
    pair = symbol.replace("-", "").replace("/", "")
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": minutes, "since": int(start.timestamp())},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"kraken: {payload['error']}")
    key = [k for k in payload["result"] if k != "last"][0]
    df = pd.DataFrame(
        payload["result"][key],
        columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    df = df[(df.index >= _ts(start)) & (df.index <= _ts(end))]
    return _normalise(df)


# ---------------------------------------------------------------- histdata (forex / metals)
_HD_PAGE = (
    "https://www.histdata.com/download-free-forex-historical-data/"
    "?/ascii/1-minute-bar-quotes/{pair}/{path}"
)
_HD_FIELDS = re.compile(r'id="(tk|date|datemonth|platform|timeframe|fxpair)" value="([^"]*)"')


def _histdata_archive(session: requests.Session, pair: str, path: str) -> pd.DataFrame | None:
    """One archive: path '2024' (past year) or '2026/5' (current-year month)."""
    page = _HD_PAGE.format(pair=pair.lower(), path=path)
    r = session.get(page, timeout=60)
    r.raise_for_status()
    fields = dict(_HD_FIELDS.findall(r.text))
    if "tk" not in fields:
        return None
    r = session.post(
        "https://www.histdata.com/get.php",
        data=fields,
        headers={"Referer": page},
        timeout=120,
    )
    r.raise_for_status()
    if r.content[:2] != b"PK":
        return None
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(
        zf.open(name), sep=";", header=None,
        names=["dt", "open", "high", "low", "close", "volume"],
    )
    df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    # HistData timestamps are US Eastern Standard Time (GMT-5, no DST) -> UTC
    df["dt"] = df["dt"].dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC")
    return df.set_index("dt")


def _histdata(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research pipeline)"
    frames = []
    for year in range(start.year, end.year + 1):
        if year < end.year:
            df = _histdata_archive(session, symbol, str(year))
            if df is not None:
                frames.append(df)
        else:  # current year comes as monthly archives
            for month in range(1, end.month + 1):
                df = _histdata_archive(session, symbol, f"{year}/{month}")
                if df is not None:
                    frames.append(df)
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame(columns=COLS)
    m1 = _normalise(pd.concat(frames))
    m1 = m1[(m1.index >= _ts(start)) & (m1.index <= _ts(end))]
    return resample(m1, timeframe)


# ---------------------------------------------------------------- csv / MT5 export
def _csv(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """`symbol` is a file path. Handles generic OHLCV CSVs and MT5 exports
    (tab-separated, <DATE>\t<TIME>\t<OPEN>... header)."""
    path = Path(symbol)
    head = path.open("r", errors="ignore").readline()
    if "<DATE>" in head:  # MetaTrader 5 export
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "00:00:00"))
        df = df.rename(columns={"tickvol": "volume"})
        if "volume" not in df or df["volume"].eq(0).all():
            df["volume"] = df.get("vol", 0)
        df.index = dt.dt.tz_localize("UTC")
    else:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        tcol = next(c for c in ("dt", "datetime", "date", "time", "timestamp") if c in df.columns)
        idx = pd.to_datetime(df[tcol], utc=True)
        df.index = idx
        if "volume" not in df.columns:
            df["volume"] = 0.0
    df = _normalise(df)
    df = df[(df.index >= _ts(start)) & (df.index <= _ts(end))]
    return resample(df, timeframe) if timeframe else df


PROVIDERS = {
    "coinbase": _coinbase,
    "bitstamp": _bitstamp,
    "kraken": _kraken,
    "histdata": _histdata,
    "csv": _csv,
}

_PANDAS_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "6h": "6h", "1d": "1D"}


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample to a coarser timeframe (no-op if bars already match)."""
    if df.empty:
        return df
    freq = _PANDAS_FREQ[timeframe]
    step = pd.tseries.frequencies.to_offset(freq).nanos
    med = df.index.to_series().diff().median()
    if pd.notna(med) and med.value >= step:
        return df
    out = df.resample(freq, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def load_ohlcv(
    provider: str,
    symbol: str,
    timeframe: str = "1h",
    years: float = 2.0,
    end: datetime | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load normalised OHLCV for `symbol` covering the last `years` years."""
    end = end or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=round(365.25 * years))
    key = re.sub(r"[^A-Za-z0-9_-]", "_", f"{provider}_{symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}")
    cache_file = CACHE / f"{key}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    df = PROVIDERS[provider](symbol, timeframe, start, end)
    if df.empty:
        raise RuntimeError(f"{provider}:{symbol} returned no data for {start:%F}..{end:%F}")
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df
