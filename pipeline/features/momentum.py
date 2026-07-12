"""Momentum: RSI/MACD, regular + hidden divergence, acceleration, exhaustion
(MARKET_RULES.md §9). Divergences compare the last two confirmed swing
points of price vs the oscillator — causal by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import find_swings


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def _divergence(df: pd.DataFrame, osc: pd.Series, k: int = 5) -> tuple[pd.Series, pd.Series]:
    """(regular, hidden) divergence flags from the last two confirmed swings.

    regular:  price LL but osc HL -> +1 (bullish);  price HH but osc LH -> -1
    hidden:   price HL but osc LL -> +1 (continuation); price LH but osc HH -> -1
    """
    sw = find_swings(df["high"], df["low"], k)
    hi_ev = sw["swing_hi_px"].where(sw["swing_hi_i"].diff().ne(0))
    lo_ev = sw["swing_lo_px"].where(sw["swing_lo_i"].diff().ne(0))

    # oscillator value at the swing bar (osc at i is known at i+k when swing confirms)
    osc_at_hi = pd.Series(np.nan, index=df.index)
    osc_at_lo = pd.Series(np.nan, index=df.index)
    hi_pos = sw["swing_hi_i"].where(sw["swing_hi_i"].diff().ne(0)).dropna().astype(int)
    lo_pos = sw["swing_lo_i"].where(sw["swing_lo_i"].diff().ne(0)).dropna().astype(int)
    osc_at_hi.loc[hi_pos.index] = osc.values[hi_pos.values]
    osc_at_lo.loc[lo_pos.index] = osc.values[lo_pos.values]

    def pair(ev, osc_ev):
        cur_p, prev_p = ev.ffill(), ev.dropna().shift(1).reindex(ev.index).ffill()
        cur_o, prev_o = osc_ev.ffill(), osc_ev.dropna().shift(1).reindex(ev.index).ffill()
        return cur_p, prev_p, cur_o, prev_o

    hp, hp0, ho, ho0 = pair(hi_ev, osc_at_hi)
    lp, lp0, lo_, lo0 = pair(lo_ev, osc_at_lo)

    regular = pd.Series(0.0, index=df.index)
    regular[(lp < lp0) & (lo_ > lo0)] = 1.0    # bullish
    regular[(hp > hp0) & (ho < ho0)] = -1.0    # bearish
    hidden = pd.Series(0.0, index=df.index)
    hidden[(lp > lp0) & (lo_ < lo0)] = 1.0
    hidden[(hp < hp0) & (ho > ho0)] = -1.0
    return regular, hidden


def compute(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    r = rsi(c)
    line, sig, hist = macd(c)
    out["mom_rsi"] = r
    out["mom_rsi_slope"] = r.diff(5)
    out["mom_macd_hist"] = hist / c
    out["mom_macd_cross"] = np.sign(line - sig)

    reg_r, hid_r = _divergence(df, r)
    reg_m, hid_m = _divergence(df, line)
    out["mom_rsi_div"] = reg_r
    out["mom_rsi_hidden_div"] = hid_r
    out["mom_macd_div"] = reg_m
    out["mom_macd_hidden_div"] = hid_m

    roc = c.pct_change(10)
    out["mom_accel"] = roc.diff(5)
    out["mom_exhaustion"] = ((r > 70) & (r.diff(5) < 0)).astype(float) - (
        (r < 30) & (r.diff(5) > 0)
    ).astype(float)
    return out
