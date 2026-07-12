"""Classical chart patterns from the confirmed swing sequence
(MARKET_RULES.md §4; cross-checked with external/TradingPatternScanner).

Detection is causal: a pattern can only fire on the bar where its final
swing is confirmed. Rules are prominence-based (extremes must stand out
from intervening swings by multiples of ATR) — tuned against
pipeline/bench/pattern_bench.py to >=95% accuracy on textbook shapes while
staying silent on random walks. Signed one-hots: +1 bullish, -1 bearish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import find_swings
from .trend import atr

# tolerances in ATR units (tuned on the benchmark; see reports/PATTERN_BENCH.md)
EQ_TOL = 1.5        # "equal" extremes differ by less than this
FLAT_TOL = 1.4      # "flat" triangle side tolerance
PROM = 4.0          # prominence: extreme vs intervening counter-swing
HEAD_MIN = 1.2      # head must exceed shoulders by this
SLOPE_MIN = 2.0     # rising/falling trendline: min move between touches
HEIGHT_MIN = 5.0    # min triangle/wedge/channel height (widest point)
FLAG_LEG = 8.0      # impulse leg size for flags


def _swing_events(df: pd.DataFrame, k: int):
    """Chronological confirmed swings: (confirm_pos, swing_pos, price, kind)."""
    sw = find_swings(df["high"], df["low"], k)
    pos = {ts: p for p, ts in enumerate(df.index)}
    events = []
    hi = sw["swing_hi_i"].where(sw["swing_hi_i"].diff().ne(0)).dropna()
    lo = sw["swing_lo_i"].where(sw["swing_lo_i"].diff().ne(0)).dropna()
    for ts, sp in hi.items():
        events.append((pos[ts], int(sp), df["high"].values[int(sp)], 1))
    for ts, sp in lo.items():
        events.append((pos[ts], int(sp), df["low"].values[int(sp)], -1))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _between(seq, p1, p2, kind):
    """Extreme price of `kind` swings strictly between swing positions p1<p2."""
    vals = [px for sp, px, kd in seq if p1 < sp < p2 and kd == kind]
    if not vals:
        return None
    return min(vals) if kind == -1 else max(vals)


def compute(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    n = len(df)
    a = atr(df).values
    events = _swing_events(df, k)

    names = ["pat_double_top", "pat_double_bottom", "pat_triple_top",
             "pat_triple_bottom", "pat_hs", "pat_inv_hs", "pat_asc_triangle",
             "pat_desc_triangle", "pat_sym_triangle", "pat_rising_wedge",
             "pat_falling_wedge", "pat_channel", "pat_flag"]
    cols = {m: np.zeros(n) for m in names}

    hist: list[tuple[int, float, int]] = []  # (swing_pos, price, kind)
    ei = 0
    for i in range(n):
        fresh = False
        while ei < len(events) and events[ei][0] <= i:
            hist.append(events[ei][1:])
            fresh = True
            ei += 1
        # fire only on the bar where a new swing confirms (kills persistence FPs)
        if not fresh or len(hist) < 4 or np.isnan(a[i]) or a[i] == 0:
            continue
        atr_i = a[i]
        seq = hist[-10:]
        highs = [(sp, px) for sp, px, kd in seq if kd == 1]
        lows = [(sp, px) for sp, px, kd in seq if kd == -1]
        if len(highs) < 2 or len(lows) < 2:
            continue

        def eq(x, y, tol=EQ_TOL):
            return abs(x - y) < tol * atr_i

        # ---- double / triple top: near-equal prominent highs
        (p1, h1), (p2, h2) = highs[-2], highs[-1]
        trough = _between(seq, p1, p2, -1)
        if trough is not None and eq(h1, h2) and min(h1, h2) - trough > PROM * atr_i:
            cols["pat_double_top"][i] = -1
            if len(highs) >= 3:
                p0, h0 = highs[-3]
                t0 = _between(seq, p0, p1, -1)
                if t0 is not None and eq(h0, h1) and min(h0, h1) - t0 > PROM * atr_i:
                    cols["pat_triple_top"][i] = -1

        # ---- double / triple bottom
        (q1, l1), (q2, l2) = lows[-2], lows[-1]
        crest = _between(seq, q1, q2, 1)
        if crest is not None and eq(l1, l2) and crest - max(l1, l2) > PROM * atr_i:
            cols["pat_double_bottom"][i] = 1
            if len(lows) >= 3:
                q0, l0 = lows[-3]
                c0 = _between(seq, q0, q1, 1)
                if c0 is not None and eq(l0, l1) and c0 - max(l0, l1) > PROM * atr_i:
                    cols["pat_triple_bottom"][i] = 1

        # ---- head & shoulders: prominent head, near-equal shoulders, real neckline
        if len(highs) >= 3:
            (pl, ls), (ph, hd), (pr, rs) = highs[-3], highs[-2], highs[-1]
            n1 = _between(seq, pl, ph, -1)
            n2 = _between(seq, ph, pr, -1)
            if (n1 is not None and n2 is not None
                    and hd - max(ls, rs) > HEAD_MIN * atr_i
                    and eq(ls, rs, 2 * EQ_TOL)
                    and min(ls, rs) - max(n1, n2) > PROM * atr_i):
                cols["pat_hs"][i] = -1
        if len(lows) >= 3:
            (ql, ls), (qh, hd), (qr, rs) = lows[-3], lows[-2], lows[-1]
            n1 = _between(seq, ql, qh, 1)
            n2 = _between(seq, qh, qr, 1)
            if (n1 is not None and n2 is not None
                    and min(ls, rs) - hd > HEAD_MIN * atr_i
                    and eq(ls, rs, 2 * EQ_TOL)
                    and min(n1, n2) - max(ls, rs) > PROM * atr_i):
                cols["pat_inv_hs"][i] = 1

        # ---- triangles / wedges / channel: need 3 monotonic touches per side
        if (len(highs) >= 3 and len(lows) >= 3
                and max(px for _, px in highs[-3:]) - min(px for _, px in lows[-3:])
                    > HEIGHT_MIN * atr_i):
            h3 = [px for _, px in highs[-3:]]
            l3 = [px for _, px in lows[-3:]]
            flat_h = eq(h3[0], h3[2], FLAT_TOL) and eq(h3[0], h3[1], FLAT_TOL)
            flat_l = eq(l3[0], l3[2], FLAT_TOL) and eq(l3[0], l3[1], FLAT_TOL)
            rising_l = l3[2] - l3[1] > 0 and l3[1] - l3[0] > 0 and l3[2] - l3[0] > SLOPE_MIN * atr_i
            falling_h = h3[2] - h3[1] < 0 and h3[1] - h3[0] < 0 and h3[0] - h3[2] > SLOPE_MIN * atr_i
            rising_h = h3[2] - h3[1] > 0 and h3[1] - h3[0] > 0 and h3[2] - h3[0] > SLOPE_MIN * atr_i
            falling_l = l3[2] - l3[1] < 0 and l3[1] - l3[0] < 0 and l3[0] - l3[2] > SLOPE_MIN * atr_i

            if flat_h and rising_l:
                cols["pat_asc_triangle"][i] = 1
            elif flat_l and falling_h:
                cols["pat_desc_triangle"][i] = -1
            elif falling_h and rising_l:
                cols["pat_sym_triangle"][i] = 0.5  # direction unknown until break
            elif rising_h and rising_l:
                conv = (h3[2] - h3[0]) < (l3[2] - l3[0]) - 0.5 * atr_i
                cols["pat_rising_wedge" if conv else "pat_channel"][i] = -1 if conv else 1
            elif falling_h and falling_l:
                conv = (h3[0] - h3[2]) < (l3[0] - l3[2]) - 0.5 * atr_i
                cols["pat_falling_wedge" if conv else "pat_channel"][i] = 1 if conv else -1

        # ---- flag: big impulse then shallow 2-swing drift against it
        if len(seq) >= 4:
            leg = (seq[-3][1] - seq[-4][1]) / atr_i
            drift = (seq[-1][1] - seq[-3][1]) / atr_i
            if abs(leg) > FLAG_LEG and abs(drift) < 0.35 * abs(leg) and np.sign(drift) != np.sign(leg):
                cols["pat_flag"][i] = np.sign(leg)

    out = pd.DataFrame(cols, index=df.index)
    # a detected pattern stays active for 2k bars for the model's benefit
    return out.replace(0.0, np.nan).ffill(limit=2 * k).fillna(0.0)
