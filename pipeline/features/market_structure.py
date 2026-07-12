"""Market structure: swings, HH/HL/LH/LL, BOS, CHoCH, MSS, swing rhythm.

Causality: a swing extreme at bar *i* with confirmation width *k* is only
*known* at bar *i+k*; every feature here uses the last swing that was
already confirmed at the current bar, so nothing peeks ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_swings(high: pd.Series, low: pd.Series, k: int) -> pd.DataFrame:
    """Confirmed swing points.

    Returns a frame indexed like the input with columns:
        swing_hi_px / swing_lo_px : price of the last confirmed swing high/low
        swing_hi_i  / swing_lo_i  : integer bar position of that swing
    Values update only at confirmation time (i+k) and forward-fill after.
    """
    h, l = high.values, low.values
    n = len(h)
    hi_px = np.full(n, np.nan)
    lo_px = np.full(n, np.nan)
    hi_i = np.full(n, -1.0)
    lo_i = np.full(n, -1.0)
    for i in range(k, n - k):
        seg_h = h[i - k : i + k + 1]
        if h[i] == seg_h.max() and (seg_h[k] > seg_h[np.arange(2 * k + 1) != k]).sum() >= 2 * k - 1:
            hi_px[i + k] = h[i]  # known only k bars later
            hi_i[i + k] = i
        seg_l = l[i - k : i + k + 1]
        if l[i] == seg_l.min() and (seg_l[k] < seg_l[np.arange(2 * k + 1) != k]).sum() >= 2 * k - 1:
            lo_px[i + k] = l[i]
            lo_i[i + k] = i
    out = pd.DataFrame(
        {"swing_hi_px": hi_px, "swing_lo_px": lo_px, "swing_hi_i": hi_i, "swing_lo_i": lo_i},
        index=high.index,
    )
    out[["swing_hi_px", "swing_lo_px"]] = out[["swing_hi_px", "swing_lo_px"]].ffill()
    out[["swing_hi_i", "swing_lo_i"]] = out[["swing_hi_i", "swing_lo_i"]].replace(-1.0, np.nan).ffill()
    return out


def structure_features(df: pd.DataFrame, k: int, prefix: str) -> pd.DataFrame:
    """HH/HL/LH/LL flags, BOS/CHoCH direction and recency, MSS, rhythm."""
    close = df["close"]
    sw = find_swings(df["high"], df["low"], k)
    n = len(df)

    hi_px, lo_px = sw["swing_hi_px"].values, sw["swing_lo_px"].values
    hi_i = sw["swing_hi_i"].values

    # sequence of confirmed swing highs/lows (as events)
    hi_event = sw["swing_hi_px"].where(sw["swing_hi_i"].diff().ne(0))
    lo_event = sw["swing_lo_px"].where(sw["swing_lo_i"].diff().ne(0))

    prev_hi = hi_event.dropna().shift(1).reindex(df.index).ffill()
    prev_lo = lo_event.dropna().shift(1).reindex(df.index).ffill()

    hh = (pd.Series(hi_px, index=df.index) > prev_hi).astype(float)
    ll = (pd.Series(lo_px, index=df.index) < prev_lo).astype(float)
    hl = (pd.Series(lo_px, index=df.index) > prev_lo).astype(float)
    lh = (pd.Series(hi_px, index=df.index) < prev_hi).astype(float)

    # trend state from swing sequence: +1 while HH+HL, -1 while LH+LL
    raw = np.where(hh + hl == 2, 1.0, np.where(lh + ll == 2, -1.0, np.nan))
    trend = pd.Series(raw, index=df.index).ffill().fillna(0.0)

    # BOS: close breaks last swing extreme WITH trend; CHoCH: against trend
    brk_up = (close > pd.Series(hi_px, index=df.index)).astype(float)
    brk_dn = (close < pd.Series(lo_px, index=df.index)).astype(float)
    new_brk_up = (brk_up.diff() == 1) & (pd.Series(hi_px, index=df.index).notna())
    new_brk_dn = (brk_dn.diff() == 1) & (pd.Series(lo_px, index=df.index).notna())

    bos_dir = pd.Series(0.0, index=df.index)
    choch_dir = pd.Series(0.0, index=df.index)
    bos_dir[new_brk_up & (trend >= 0)] = 1.0
    bos_dir[new_brk_dn & (trend <= 0)] = -1.0
    choch_dir[new_brk_up & (trend < 0)] = 1.0
    choch_dir[new_brk_dn & (trend > 0)] = -1.0

    def bars_since(events: pd.Series) -> pd.Series:
        idx = np.arange(n, dtype=float)
        marks = np.where(events.values != 0, idx, np.nan)
        last = pd.Series(marks, index=df.index).ffill()
        return (pd.Series(idx, index=df.index) - last).fillna(n).clip(0, 500)

    # MSS: CHoCH followed by structure agreeing with the new direction
    last_choch = choch_dir.replace(0.0, np.nan).ffill().fillna(0.0)
    mss = ((last_choch == 1.0) & (trend == 1.0)).astype(float) - (
        (last_choch == -1.0) & (trend == -1.0)
    ).astype(float)

    # swing rhythm: current leg direction, pullback depth of the counter-move
    rng = pd.Series(hi_px - lo_px, index=df.index)
    leg_dir = np.where(pd.Series(hi_i, index=df.index) > sw["swing_lo_i"], -1.0, 1.0)
    # depth of retracement from the leg extreme, as a fraction of the leg
    retr_up = (pd.Series(hi_px, index=df.index) - close) / rng.replace(0, np.nan)
    retr_dn = (close - pd.Series(lo_px, index=df.index)) / rng.replace(0, np.nan)
    pullback = pd.Series(np.where(leg_dir > 0, retr_up, retr_dn), index=df.index).clip(-1, 2)

    # V-turn sharpness: |return over k bars| flip magnitude at the last swing
    ret_k = close.pct_change(k)
    v_sharp = (ret_k - ret_k.shift(k)).abs()

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_hh"] = hh
    out[f"{prefix}_hl"] = hl
    out[f"{prefix}_lh"] = lh
    out[f"{prefix}_ll"] = ll
    out[f"{prefix}_trend"] = trend
    out[f"{prefix}_bos_dir"] = bos_dir.replace(0.0, np.nan).ffill().fillna(0.0)
    out[f"{prefix}_bars_since_bos"] = bars_since(bos_dir)
    out[f"{prefix}_choch_dir"] = last_choch
    out[f"{prefix}_bars_since_choch"] = bars_since(choch_dir)
    out[f"{prefix}_mss"] = mss
    out[f"{prefix}_leg_dir"] = leg_dir
    out[f"{prefix}_pullback_depth"] = pullback
    out[f"{prefix}_v_sharpness"] = v_sharp
    out[f"{prefix}_dist_swing_hi"] = (close - hi_px) / close
    out[f"{prefix}_dist_swing_lo"] = (close - lo_px) / close
    return out


def compute(df: pd.DataFrame, k_internal: int = 3, k_external: int = 8) -> pd.DataFrame:
    """Internal (minor) + external (major) structure features."""
    internal = structure_features(df, k_internal, "ms_int")
    external = structure_features(df, k_external, "ms_ext")
    return pd.concat([internal, external], axis=1)
