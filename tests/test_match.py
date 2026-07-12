"""Tests for the live pattern-match engine: regime conditioning separates a
signal that works only in bull regimes; the snapshot flips its verdict
bar-by-bar when opposing patterns fire; ledger stats count events once."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.match import (PATTERN_SIGNALS, conditional_stats, live_snapshot,
                            regime_series)


def _frame(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(0.003 * rng.standard_normal(n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.roll(close, 1), "high": close * 1.004,
                       "low": close * 0.996, "close": close, "volume": 1.0},
                      index=idx)
    df.iloc[0, 0] = close[0]
    return df


def test_regime_conditional_hit_rates():
    """Plant a signal that predicts correctly ONLY in the bull regime."""
    rng = np.random.default_rng(1)
    n = 2000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    trend = np.where(np.arange(n) < n // 2, 1.0, -1.0)  # bull half then bear half
    ret = 0.002 * trend + 0.001 * rng.standard_normal(n)
    close = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)

    X = pd.DataFrame(index=idx)
    X["trend_regime"] = trend  # drives regime_series
    sig = np.zeros(n)
    ev = rng.choice(np.arange(50, n - 50), 200, replace=False)
    sig[ev] = 1.0  # always-bullish signal
    X["pat_double_bottom"] = sig

    stats, reg = conditional_stats(X, close, horizon=24)
    row_bull = stats[(stats.signal == "pat_double_bottom") & (stats.regime == 2)]
    row_bear = stats[(stats.signal == "pat_double_bottom") & (stats.regime == 0)]
    assert row_bull.n.iloc[0] > 20 and row_bear.n.iloc[0] > 20
    assert row_bull.hit_rate.iloc[0] > 0.9      # works in bull
    assert row_bear.hit_rate.iloc[0] < 0.1      # fails in bear — kept separate


def test_snapshot_flips_with_live_pattern():
    df = _frame()
    idx = df.index
    X = pd.DataFrame(index=idx)
    X["trend_regime"] = 0.0
    for col in ("pat_double_bottom", "pat_double_top"):
        X[col] = 0.0
    # history: bullish signal reliable, bearish signal reliable (both n>=10)
    rng = np.random.default_rng(2)
    up_ev = rng.choice(np.arange(50, 1100, 2), 40, replace=False)
    dn_ev = rng.choice(np.arange(51, 1100, 2), 40, replace=False)
    X.loc[idx[up_ev], "pat_double_bottom"] = 1.0
    X.loc[idx[dn_ev], "pat_double_top"] = -1.0
    close = df.close.copy()
    # force outcomes so both signals are ~always right historically
    fwd = pd.Series(0.0, index=idx)
    stats, reg = conditional_stats(X, close, horizon=24)
    # hand the snapshot a doctored ledger where both patterns are 90% accurate
    stats.loc[stats.signal == "pat_double_bottom", ["n", "hit_rate"]] = [40, 0.9]
    stats.loc[stats.signal == "pat_double_top", ["n", "hit_rate"]] = [40, 0.9]

    X_bull = X.copy()
    X_bull.iloc[-1, X_bull.columns.get_loc("pat_double_bottom")] = 1.0
    snap = live_snapshot(df, X_bull, stats, reg)
    assert snap["verdict"] == "CALL"
    assert snap["trade_plan"]["tp"] > snap["trade_plan"]["sl"]

    X_bear = X.copy()
    X_bear.iloc[-1, X_bear.columns.get_loc("pat_double_top")] = -1.0
    snap2 = live_snapshot(df, X_bear, stats, reg)
    assert snap2["verdict"] == "PUT"     # verdict flips on the very next state


def test_events_counted_once():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, 300), index=idx)
    X = pd.DataFrame(index=idx)
    X["trend_regime"] = 1.0
    v = np.zeros(300)
    v[100:110] = 1.0  # one activation persisting 10 bars = ONE event
    X["smc_sweep_dir"] = v
    stats, _ = conditional_stats(X, close, horizon=10)
    n_total = stats[stats.signal == "smc_sweep_dir"].n.sum()
    assert n_total == 1
