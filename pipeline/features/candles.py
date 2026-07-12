"""Candlestick geometry + named patterns from geometry (MARKET_RULES.md §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    up_wick = h - np.maximum(o, c)
    dn_wick = np.minimum(o, c) - l

    out = pd.DataFrame(index=df.index)
    # geometry (the model's raw vocabulary)
    out["cdl_body_frac"] = body / rng
    out["cdl_up_wick_frac"] = up_wick / rng
    out["cdl_dn_wick_frac"] = dn_wick / rng
    out["cdl_close_pos"] = (c - l) / rng           # close position within range
    out["cdl_dir"] = np.sign(c - o)
    out["cdl_gap"] = (o - c.shift()) / c.shift()
    out["cdl_rel_range"] = rng / rng.rolling(20).mean()

    bull = c > o
    bear = c < o
    small_body = out["cdl_body_frac"] < 0.3
    big_body = out["cdl_body_frac"] > 0.7

    # named patterns, signed where directional (+1 bullish, -1 bearish)
    out["cdl_doji"] = (out["cdl_body_frac"] < 0.1).astype(float)
    out["cdl_hammer"] = (small_body & (out["cdl_dn_wick_frac"] > 0.6)).astype(float)
    out["cdl_shooting_star"] = -(small_body & (out["cdl_up_wick_frac"] > 0.6)).astype(float)
    out["cdl_marubozu"] = (out["cdl_body_frac"] > 0.9).astype(float) * out["cdl_dir"]
    out["cdl_spinning_top"] = (
        small_body & (out["cdl_up_wick_frac"] > 0.25) & (out["cdl_dn_wick_frac"] > 0.25)
    ).astype(float)

    engulf_bull = bull & bear.shift(1, fill_value=False) & (c >= o.shift()) & (o <= c.shift())
    engulf_bear = bear & bull.shift(1, fill_value=False) & (c <= o.shift()) & (o >= c.shift())
    out["cdl_engulfing"] = engulf_bull.astype(float) - engulf_bear.astype(float)

    inside = (np.maximum(o, c) <= np.maximum(o, c).shift()) & (
        np.minimum(o, c) >= np.minimum(o, c).shift()
    )
    out["cdl_harami"] = (inside & small_body).astype(float) * out["cdl_dir"]

    # morning/evening star: big move, pause, reversal past midpoint of bar1
    mid1 = (o.shift(2) + c.shift(2)) / 2
    star_dn = big_body.shift(2, fill_value=False) & bear.shift(2, fill_value=False) & small_body.shift(1, fill_value=False)
    star_up = big_body.shift(2, fill_value=False) & bull.shift(2, fill_value=False) & small_body.shift(1, fill_value=False)
    out["cdl_star"] = (star_dn & bull & (c > mid1)).astype(float) - (
        star_up & bear & (c < mid1)
    ).astype(float)

    three_up = (bull & bull.shift(1, fill_value=False) & bull.shift(2, fill_value=False)
                & (c > c.shift()) & (c.shift() > c.shift(2)))
    three_dn = (bear & bear.shift(1, fill_value=False) & bear.shift(2, fill_value=False)
                & (c < c.shift()) & (c.shift() < c.shift(2)))
    out["cdl_three_soldiers_crows"] = three_up.astype(float) - three_dn.astype(float)
    return out
