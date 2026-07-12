"""Phase 2 tests: shape/causality checks per feature family on synthetic data
with a known structure, plus a no-lookahead test (appending future bars must
not change past feature values)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features import build_features, FEATURE_GROUPS
from pipeline.features.market_structure import find_swings


def synth(n=800, seed=7, freq="1h"):
    rng = np.random.default_rng(seed)
    # trending random walk with cycles so swings/patterns exist
    drift = np.concatenate([np.full(n // 2, 0.0004), np.full(n - n // 2, -0.0004)])
    ret = drift + 0.004 * rng.standard_normal(n) + 0.002 * np.sin(np.arange(n) / 15)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(0.002 * rng.standard_normal(n)))
    low = close * (1 - np.abs(0.002 * rng.standard_normal(n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = np.abs(rng.normal(1000, 300, n)) * (1 + np.abs(ret) * 100)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(high, close), "low": np.minimum(low, close),
         "close": close, "volume": vol},
        index=idx,
    )


DF = synth()


def test_swings_confirmed_causally():
    sw = find_swings(DF.high, DF.low, k=5)
    # swing index recorded at confirmation must be >= k bars in the past
    conf = sw["swing_hi_i"].dropna()
    pos = pd.Series(range(len(DF)), index=DF.index)
    assert ((pos.loc[conf.index] - conf) >= 5).all()


@pytest.mark.parametrize("group", list(FEATURE_GROUPS))
def test_group_shapes(group):
    feats = FEATURE_GROUPS[group](DF)
    assert len(feats) == len(DF)
    assert feats.index.equals(DF.index)
    assert feats.shape[1] >= 2
    # everything numeric, no infs
    vals = feats.to_numpy(dtype=float)
    assert not np.isinf(vals[~np.isnan(vals)]).any()


def test_build_features_matrix():
    X = build_features(DF)
    assert len(X) == len(DF) - 220
    assert X.shape[1] > 80
    # non-options features should be mostly populated after warmup
    non_opt = [c for c in X.columns if not c.startswith("opt_")]
    assert X[non_opt].notna().mean().mean() > 0.85


def test_no_lookahead():
    """Feature values at bar t must be identical whether or not future bars exist."""
    cut = 600
    X_full = build_features(DF, warmup=220)
    X_trunc = build_features(DF.iloc[:cut], warmup=220)
    common = X_trunc.index
    a = X_full.loc[common].fillna(-999).to_numpy(dtype=float)
    b = X_trunc.fillna(-999).to_numpy(dtype=float)
    mismatch = np.abs(a - b) > 1e-9
    frac = mismatch.mean()
    assert frac == 0.0, f"lookahead detected in {X_trunc.columns[mismatch.any(0)].tolist()}"


def test_daily_bars_drop_session_features():
    d = synth(freq="1D")
    feats = FEATURE_GROUPS["sessions"](d)
    assert feats["ses_asia"].isna().all()
    assert feats["ses_dow"].notna().all()
