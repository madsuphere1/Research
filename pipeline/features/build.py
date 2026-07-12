"""Assemble every feature family into one matrix (causal end to end)."""

from __future__ import annotations

import pandas as pd

from . import (
    candles,
    classical,
    market_structure,
    momentum,
    options_stub,
    sessions,
    smc,
    support_resistance,
    timeseries,
    trend,
    volatility,
    volume,
)

FEATURE_GROUPS = {
    "market_structure": market_structure.compute,
    "trend": trend.compute,
    "smc": smc.compute,
    "classical": classical.compute,
    "candles": candles.compute,
    "volume": volume.compute,
    "volatility": volatility.compute,
    "momentum": momentum.compute,
    "support_resistance": support_resistance.compute,
    "sessions": sessions.compute,
    "timeseries": timeseries.compute,
    "options": options_stub.compute,
}


def build_features(
    df: pd.DataFrame,
    groups: list[str] | None = None,
    warmup: int = 220,
) -> pd.DataFrame:
    """Feature matrix aligned to df.index, first `warmup` bars dropped
    (longest indicator lookback is the 200-EMA / 120-bar profile)."""
    parts = []
    for name, fn in FEATURE_GROUPS.items():
        if groups is not None and name not in groups:
            continue
        feats = fn(df)
        assert feats.index.equals(df.index), f"{name}: index mismatch"
        parts.append(feats)
    X = pd.concat(parts, axis=1)
    if len(X) > warmup:
        X = X.iloc[warmup:]
    return X


def build_features_cached(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """build_features with an on-disk cache keyed by the data's content hash.

    The feature matrix is deterministic for a given OHLCV frame, so repeat
    runs (simulate/match/scan/run on the same cached download) skip the
    expensive Python loops entirely.
    """
    import hashlib
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parents[2] / "data_cache"
    h = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]
    path = cache_dir / f"feat_{h}_{kw.get('warmup', 220)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    X = build_features(df, **kw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path)
    return X
