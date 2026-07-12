"""Volatility patterns: ATR/Bollinger squeeze & expansion (MARKET_RULES.md §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import atr


def compute(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    a14 = atr(df)
    out["vlt_atr_pct"] = a14 / c
    out["vlt_atr_ratio"] = (a14 / a14.rolling(50).mean()).clip(0, 5)
    out["vlt_atr_expanding"] = (a14 > a14.shift(10)).astype(float)

    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bw = (4 * sd / ma).replace(0, np.nan)  # Bollinger bandwidth
    out["vlt_bb_width"] = bw
    # squeeze: bandwidth in its lowest quintile of the last 120 bars
    out["vlt_bb_squeeze"] = (bw <= bw.rolling(120).quantile(0.2)).astype(float)
    out["vlt_bb_expansion"] = (bw >= bw.rolling(120).quantile(0.8)).astype(float)
    # squeeze release direction: first close outside the bands after a squeeze
    upper, lower = ma + 2 * sd, ma - 2 * sd
    was_squeezed = out["vlt_bb_squeeze"].shift(1).rolling(5).max()
    out["vlt_squeeze_release"] = np.where(
        (was_squeezed > 0) & (c > upper), 1.0, np.where((was_squeezed > 0) & (c < lower), -1.0, 0.0)
    )

    rv = c.pct_change().rolling(20).std()
    out["vlt_rv_regime"] = (rv / rv.rolling(120).median()).clip(0, 5)
    return out
