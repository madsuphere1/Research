"""Classical chart patterns from the confirmed swing sequence
(MARKET_RULES.md §4; approach cross-checked with external/TradingPatternScanner).

Patterns are detected on the last 5-7 confirmed swing points, so detection
is causal (a pattern exists only once its final swing is confirmed).
Signed one-hots: +1 bullish implication, -1 bearish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import find_swings
from .trend import atr


def _swing_sequence(df: pd.DataFrame, k: int):
    """Chronological (bar_position, price, kind) of confirmed swings; kind +1 high / -1 low."""
    sw = find_swings(df["high"], df["low"], k)
    hi = sw["swing_hi_i"].where(sw["swing_hi_i"].diff().ne(0)).dropna()
    lo = sw["swing_lo_i"].where(sw["swing_lo_i"].diff().ne(0)).dropna()
    events = []  # (confirm_pos, swing_pos, price, kind)
    pos = {ts: p for p, ts in enumerate(df.index)}
    for ts, sp in hi.items():
        events.append((pos[ts], int(sp), df["high"].values[int(sp)], 1))
    for ts, sp in lo.items():
        events.append((pos[ts], int(sp), df["low"].values[int(sp)], -1))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def compute(df: pd.DataFrame, k: int = 5, tol: float = 0.5) -> pd.DataFrame:
    n = len(df)
    a = atr(df).values
    events = _swing_sequence(df, k)

    cols = {
        "pat_double_top": np.zeros(n), "pat_double_bottom": np.zeros(n),
        "pat_triple_top": np.zeros(n), "pat_triple_bottom": np.zeros(n),
        "pat_hs": np.zeros(n), "pat_inv_hs": np.zeros(n),
        "pat_asc_triangle": np.zeros(n), "pat_desc_triangle": np.zeros(n),
        "pat_sym_triangle": np.zeros(n), "pat_rising_wedge": np.zeros(n),
        "pat_falling_wedge": np.zeros(n), "pat_channel": np.zeros(n),
        "pat_flag": np.zeros(n),
    }

    hist: list[tuple[int, float, int]] = []  # (swing_pos, price, kind)
    ei = 0
    for i in range(n):
        while ei < len(events) and events[ei][0] <= i:
            hist.append(events[ei][1:])
            ei += 1
        if len(hist) < 5 or np.isnan(a[i]) or a[i] == 0:
            continue
        tol_px = tol * a[i]
        highs = [(p, px) for p, px, kd in hist[-8:] if kd == 1]
        lows = [(p, px) for p, px, kd in hist[-8:] if kd == -1]
        if len(highs) < 2 or len(lows) < 2:
            continue
        h_px = [px for _, px in highs]
        l_px = [px for _, px in lows]

        # double / triple top-bottom
        if abs(h_px[-1] - h_px[-2]) < tol_px:
            cols["pat_double_top"][i] = -1
            if len(h_px) >= 3 and abs(h_px[-2] - h_px[-3]) < tol_px:
                cols["pat_triple_top"][i] = -1
        if abs(l_px[-1] - l_px[-2]) < tol_px:
            cols["pat_double_bottom"][i] = 1
            if len(l_px) >= 3 and abs(l_px[-2] - l_px[-3]) < tol_px:
                cols["pat_triple_bottom"][i] = 1

        # head & shoulders: middle high above two similar shoulders
        if len(h_px) >= 3:
            ls, hd, rs = h_px[-3], h_px[-2], h_px[-1]
            if hd > ls + tol_px and hd > rs + tol_px and abs(ls - rs) < 2 * tol_px:
                cols["pat_hs"][i] = -1
        if len(l_px) >= 3:
            ls, hd, rs = l_px[-3], l_px[-2], l_px[-1]
            if hd < ls - tol_px and hd < rs - tol_px and abs(ls - rs) < 2 * tol_px:
                cols["pat_inv_hs"][i] = 1

        # trendline slopes over last 3 highs / lows (ATR units per swing)
        if len(h_px) >= 3 and len(l_px) >= 3:
            hs_ = (h_px[-1] - h_px[-3]) / (2 * a[i])
            lsl = (l_px[-1] - l_px[-3]) / (2 * a[i])
            flat_h, flat_l = abs(hs_) < tol, abs(lsl) < tol
            if flat_h and lsl > tol:
                cols["pat_asc_triangle"][i] = 1
            elif flat_l and hs_ < -tol:
                cols["pat_desc_triangle"][i] = -1
            elif hs_ < -tol and lsl > tol:
                cols["pat_sym_triangle"][i] = np.sign(hs_ + lsl)
            elif hs_ > tol and lsl > tol:
                cols["pat_rising_wedge" if hs_ < lsl else "pat_channel"][i] = -1 if hs_ < lsl else 1
            elif hs_ < -tol and lsl < -tol:
                cols["pat_falling_wedge" if hs_ > lsl else "pat_channel"][i] = 1 if hs_ > lsl else -1

        # flag: strong prior leg + shallow counter-drift
        leg = (hist[-1][1] - hist[-4][1]) / a[i]
        drift = (h_px[-1] - h_px[-2] + l_px[-1] - l_px[-2]) / (2 * a[i])
        if abs(leg) > 6 and abs(drift) < 1 and np.sign(drift) != np.sign(leg):
            cols["pat_flag"][i] = np.sign(leg)

    out = pd.DataFrame(cols, index=df.index)
    # patterns stay "active" for a while after detection
    return out.replace(0.0, np.nan).ffill(limit=2 * k).fillna(0.0)
