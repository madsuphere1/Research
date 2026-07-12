"""Smart Money Concepts (MARKET_RULES.md §5).

FVG, order blocks, breaker, liquidity sweeps of equal highs/lows,
premium/discount position, OTE zone. Own vectorised implementations,
cross-checked conceptually against external/smart-money-concepts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import find_swings
from .trend import atr


def compute(df: pd.DataFrame, k: int = 8, eq_tol_atr: float = 0.25) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(df)
    out = pd.DataFrame(index=df.index)

    # ---- Fair Value Gap: 3-bar displacement gap, tracked until filled
    bull_gap_lo = h.shift(2)  # gap between bar1 high and bar3 low
    bull_fvg = (l > bull_gap_lo) & (c.shift(1) > o.shift(1))
    bear_gap_hi = l.shift(2)
    bear_fvg = (h < bear_gap_hi) & (c.shift(1) < o.shift(1))

    fvg_dir = pd.Series(np.where(bull_fvg, 1.0, np.where(bear_fvg, -1.0, np.nan)), index=df.index)
    # gap midpoint at creation, carried forward until price trades back through it
    gap_mid = pd.Series(
        np.where(bull_fvg, (l + bull_gap_lo) / 2, np.where(bear_fvg, (h + bear_gap_hi) / 2, np.nan)),
        index=df.index,
    ).ffill()
    filled = ((fvg_dir.ffill() > 0) & (l <= gap_mid)) | ((fvg_dir.ffill() < 0) & (h >= gap_mid))
    out["smc_fvg_dir"] = fvg_dir.ffill().where(~filled, 0.0).fillna(0.0)
    out["smc_fvg_dist"] = ((c - gap_mid) / a).clip(-20, 20)

    # ---- Order block: last opposite candle before a k-swing break (displacement)
    sw = find_swings(h, l, k)
    brk_up = (c > sw["swing_hi_px"]) & (c.shift(1) <= sw["swing_hi_px"].shift(1))
    brk_dn = (c < sw["swing_lo_px"]) & (c.shift(1) >= sw["swing_lo_px"].shift(1))
    bear_candle = (c < o).rolling(5).max().shift(1)  # was there a down candle recently
    bull_candle = (c > o).rolling(5).max().shift(1)
    ob_dir = pd.Series(
        np.where(brk_up & (bear_candle > 0), 1.0, np.where(brk_dn & (bull_candle > 0), -1.0, np.nan)),
        index=df.index,
    )
    # OB zone price = low (bull) / high (bear) of the 5 bars before the break
    ob_px = pd.Series(
        np.where(brk_up, l.rolling(5).min().shift(1), np.where(brk_dn, h.rolling(5).max().shift(1), np.nan)),
        index=df.index,
    ).ffill()
    ob_dir_ff = ob_dir.ffill().fillna(0.0)
    out["smc_ob_dir"] = ob_dir_ff
    out["smc_ob_dist"] = ((c - ob_px) / a).clip(-20, 20)

    # ---- Breaker: order block violated (close through it) then price returns
    violated = ((ob_dir_ff > 0) & (c < ob_px)) | ((ob_dir_ff < 0) & (c > ob_px))
    out["smc_breaker"] = violated.astype(float) * -ob_dir_ff

    # ---- Equal highs/lows: two confirmed swing extremes within tolerance
    hi_ev = sw["swing_hi_px"].where(sw["swing_hi_i"].diff().ne(0))
    lo_ev = sw["swing_lo_px"].where(sw["swing_lo_i"].diff().ne(0))
    prev_hi = hi_ev.dropna().shift(1).reindex(df.index).ffill()
    prev_lo = lo_ev.dropna().shift(1).reindex(df.index).ffill()
    eqh = ((sw["swing_hi_px"] - prev_hi).abs() < eq_tol_atr * a).astype(float)
    eql = ((sw["swing_lo_px"] - prev_lo).abs() < eq_tol_atr * a).astype(float)
    out["smc_eqh"] = eqh
    out["smc_eql"] = eql

    # ---- Liquidity sweep: wick beyond swing extreme, close back inside
    sweep_up = (h > sw["swing_hi_px"]) & (c < sw["swing_hi_px"])  # buy-side grab
    sweep_dn = (l < sw["swing_lo_px"]) & (c > sw["swing_lo_px"])  # sell-side grab
    sweep = pd.Series(np.where(sweep_up, -1.0, np.where(sweep_dn, 1.0, 0.0)), index=df.index)
    out["smc_sweep_dir"] = sweep.replace(0.0, np.nan).ffill().fillna(0.0)
    idx = np.arange(len(df), dtype=float)
    marks = pd.Series(np.where(sweep != 0, idx, np.nan), index=df.index).ffill()
    out["smc_bars_since_sweep"] = (pd.Series(idx, index=df.index) - marks).fillna(len(df)).clip(0, 500)

    # internal sweep (inducement proxy): same on minor swings
    sw3 = find_swings(h, l, 3)
    int_sweep = pd.Series(
        np.where((h > sw3["swing_hi_px"]) & (c < sw3["swing_hi_px"]), -1.0,
                 np.where((l < sw3["swing_lo_px"]) & (c > sw3["swing_lo_px"]), 1.0, 0.0)),
        index=df.index,
    )
    out["smc_int_sweep"] = int_sweep

    # ---- Premium / discount within the external swing range
    rng = (sw["swing_hi_px"] - sw["swing_lo_px"]).replace(0, np.nan)
    pos = (c - sw["swing_lo_px"]) / rng
    out["smc_range_pos"] = pos.clip(-0.5, 1.5)  # >0.5 premium, <0.5 discount

    # ---- OTE: price inside the 0.62-0.79 retracement of the last leg
    out["smc_ote_zone"] = ((pos >= 0.21) & (pos <= 0.38)).astype(float) - (
        (pos >= 0.62) & (pos <= 0.79)
    ).astype(float)
    return out
