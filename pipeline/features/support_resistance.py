"""Support/resistance: swing-cluster levels, dynamic S/R, pivots, flip zones
(MARKET_RULES.md §10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import find_swings
from .trend import atr, ema


def compute(df: pd.DataFrame, k: int = 5, lookback: int = 250, tol_atr: float = 0.5) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    a = atr(df)
    out = pd.DataFrame(index=df.index)

    sw = find_swings(h, l, k)
    hi_ev = sw["swing_hi_px"].where(sw["swing_hi_i"].diff().ne(0))
    lo_ev = sw["swing_lo_px"].where(sw["swing_lo_i"].diff().ne(0))

    # distance to nearest clustered level above (resistance) and below (support),
    # and how many swing touches that level has (strength)
    n = len(df)
    res_d = np.full(n, np.nan)
    sup_d = np.full(n, np.nan)
    res_str = np.full(n, np.nan)
    sup_str = np.full(n, np.nan)
    levels: list[float] = []  # all confirmed swing prices so far (rolling window)
    hi_vals, lo_vals = hi_ev.values, lo_ev.values
    cv, av = c.values, a.values
    buf: list[float] = []
    for i in range(n):
        if not np.isnan(hi_vals[i]):
            buf.append(hi_vals[i])
        if not np.isnan(lo_vals[i]):
            buf.append(lo_vals[i])
        if len(buf) > 60:
            buf = buf[-60:]
        if not buf or np.isnan(av[i]) or av[i] == 0:
            continue
        arr = np.asarray(buf)
        above = arr[arr >= cv[i]]
        below = arr[arr <= cv[i]]
        if len(above):
            lvl = above.min()
            res_d[i] = (lvl - cv[i]) / av[i]
            res_str[i] = (np.abs(arr - lvl) < tol_atr * av[i]).sum()
        if len(below):
            lvl = below.max()
            sup_d[i] = (cv[i] - lvl) / av[i]
            sup_str[i] = (np.abs(arr - lvl) < tol_atr * av[i]).sum()
    out["sr_res_dist"] = pd.Series(res_d, index=df.index).clip(0, 30)
    out["sr_sup_dist"] = pd.Series(sup_d, index=df.index).clip(0, 30)
    out["sr_res_strength"] = res_str
    out["sr_sup_strength"] = sup_str

    # dynamic S/R: distance to EMAs in ATR units
    out["sr_ema50_dist"] = ((c - ema(c, 50)) / a).clip(-20, 20)
    out["sr_ema200_dist"] = ((c - ema(c, 200)) / a).clip(-20, 20)

    # classic daily-style pivot on a rolling basis
    piv = (h.rolling(24).max() + l.rolling(24).min() + c) / 3
    out["sr_pivot_dist"] = ((c - piv.shift(1)) / a).clip(-20, 20)

    # flip zone: price crossed a level recently and is retesting from the other side
    crossed_res = (c.shift(5) < c) & (out["sr_sup_dist"] < 1.0) & (out["sr_sup_strength"] >= 2)
    crossed_sup = (c.shift(5) > c) & (out["sr_res_dist"] < 1.0) & (out["sr_res_strength"] >= 2)
    out["sr_flip"] = crossed_res.astype(float) - crossed_sup.astype(float)
    return out
