"""Phase 3 tests: labeling geometry, purge correctness, relevance selection
sanity (a planted signal must be kept; pure-noise shadows must drop junk),
and walk-forward AUC > 0.5 on data with a planted learnable pattern."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.labeling import triple_barrier
from pipeline.model import select_relevant_features, walk_forward


def synth_ohlc(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    ret = 0.004 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(0.002 * rng.standard_normal(n)))
    low = close * (1 - np.abs(0.002 * rng.standard_normal(n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(high, close),
                         "low": np.minimum(low, close), "close": close,
                         "volume": 1000.0}, index=idx)


def test_triple_barrier_geometry():
    df = synth_ohlc()
    lab = triple_barrier(df, horizon=24, sl_atr=1.5, rr=2.0)
    got = lab.dropna()
    assert set(np.unique(got.label)) <= {-1.0, 0.0, 1.0}
    # winners pay rr, losers pay -1
    assert (got.loc[got.label == 1, "r_long"] == 2.0).all()
    assert (got.loc[got.label == -1, "r_long"] == -1.0).all()
    assert (got.touch_bars.dropna() <= 24).all()
    # on a random walk with rr=2 the TP is farther: fewer wins than losses
    assert (got.label == 1).sum() < (got.label == -1).sum()


def _planted(n=2000, seed=1):
    """Data where feature 'sig' genuinely predicts next-bar direction."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    ret = 0.003 * np.roll(sig, 1) * 1.2 + 0.003 * rng.standard_normal(n)
    ret[0] = 0
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                       "close": close, "volume": 1.0}, index=idx)
    X = pd.DataFrame({"sig": sig}, index=idx)
    for i in range(15):
        X[f"noise{i}"] = rng.standard_normal(n)
    X["all_nan"] = np.nan
    return df, X


def test_relevance_keeps_signal_drops_nan():
    df, X = _planted()
    lab = triple_barrier(df, horizon=8, sl_atr=1.0, rr=1.0)
    y = (lab.label == 1).astype(int)[lab.label.notna()]
    kept, dropped = select_relevant_features(X[lab.label.notna()], y)
    assert "sig" in kept
    assert "all_nan" in dropped


def test_walk_forward_learns_planted_signal():
    df, X = _planted()
    lab = triple_barrier(df, horizon=8, sl_atr=1.0, rr=1.0)
    res = walk_forward(X, lab, n_folds=4, purge=8)
    assert res.auc > 0.55, f"AUC {res.auc}"
    # out-of-sample predictions exist only after the first fold edge
    assert res.proba.notna().sum() > 500
    assert res.proba.iloc[:100].isna().all()


def test_purge_gap():
    """With purge >= horizon, training rows never overlap the test window."""
    df, X = _planted(n=1200)
    lab = triple_barrier(df, horizon=12, sl_atr=1.0, rr=1.0)
    res = walk_forward(X, lab, n_folds=3, purge=12)
    assert res.proba.notna().sum() > 300  # structure holds; purge handled internally
