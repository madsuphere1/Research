"""Phase 1 tests: providers return normalised frames; windows slice correctly.

Network tests hit only the endpoints verified reachable from this
environment (coinbase, bitstamp). Run: python -m pytest tests/test_data.py -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data import load_ohlcv, slice_window, window_permutations
from pipeline.data.providers import COLS, resample
from pipeline.data.windows import Window


def _check_normalised(df):
    assert list(df.columns) == COLS
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert (df.high >= df.low).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_window_permutations():
    wins = window_permutations(lookbacks=(6, 12, 24), offsets=(0, 6, 12), max_span_months=24)
    names = {w.name for w in wins}
    assert "lb6m_off0m" in names and "lb12m_off12m" in names
    assert "lb24m_off6m" not in names  # exceeds span
    assert all(w.lookback_months + w.offset_months <= 24 for w in wins)


def test_slice_window_synthetic():
    idx = pd.date_range("2024-01-01", "2026-01-01", freq="1D", tz="UTC")
    df = pd.DataFrame({c: np.linspace(1, 2, len(idx)) for c in COLS}, index=idx)
    w = slice_window(df, Window(6, 0))
    assert w.index.max() == df.index.max()
    assert 175 <= len(w) <= 190
    w2 = slice_window(df, Window(6, 6))
    assert w2.index.max() < w.index.min() + pd.Timedelta(days=2)


def test_resample():
    idx = pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx
    )
    d = resample(df, "1d")
    assert len(d) == 2 and d.volume.iloc[0] == 24


def test_csv_provider(tmp_path):
    idx = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    src = pd.DataFrame(
        {"datetime": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0}
    )
    p = tmp_path / "x.csv"
    src.to_csv(p, index=False)
    df = load_ohlcv("csv", str(p), "1h", years=1, end=datetime(2025, 6, 1, tzinfo=timezone.utc), use_cache=False)
    _check_normalised(df)
    assert len(df) == 100


@pytest.mark.network
def test_coinbase_live():
    df = load_ohlcv("coinbase", "BTC-USD", "1d", years=0.2, use_cache=False)
    _check_normalised(df)
    assert len(df) > 50


@pytest.mark.network
def test_bitstamp_live():
    df = load_ohlcv("bitstamp", "BTC-USD", "1d", years=0.2, use_cache=False)
    _check_normalised(df)
    assert len(df) > 50
