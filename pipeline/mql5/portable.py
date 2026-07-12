"""The 12 portable features: every one has an exact MQL5 twin in template.mq5
(same EMA/RSI/ATR recursions as pandas ewm(adjust=False); initialisation
differences wash out well inside the EA's 400-bar warmup).

Order here defines coefficient order in the generated EA — do not reorder
without regenerating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PORTABLE_ORDER = [
    "p_ema_align", "p_ema50_slope", "p_dist_e50", "p_dist_e200",
    "p_rsi", "p_rsi_slope", "p_macd_hist", "p_atr_ratio",
    "p_bb_width", "p_donch_pos", "p_ret5_atr", "p_body",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute_portable(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    a = _atr(df)
    r = _rsi(c)
    macd = _ema(c, 12) - _ema(c, 26)
    sig = macd.ewm(span=9, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)  # MQL5 twin uses population std
    dc_hi, dc_lo = h.rolling(20).max(), l.rolling(20).min()
    rng = (h - l).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["p_ema_align"] = np.sign(e20 - e50) + np.sign(e50 - e200)
    out["p_ema50_slope"] = (e50 - e50.shift(10)) / a
    out["p_dist_e50"] = (c - e50) / a
    out["p_dist_e200"] = (c - e200) / a
    out["p_rsi"] = r
    out["p_rsi_slope"] = r - r.shift(5)
    out["p_macd_hist"] = (macd - sig) / c * 10000
    out["p_atr_ratio"] = a / a.rolling(50).mean()
    out["p_bb_width"] = 4 * std20 / sma20 * 100
    out["p_donch_pos"] = (c - dc_lo) / (dc_hi - dc_lo).replace(0, np.nan)
    out["p_ret5_atr"] = (c - c.shift(5)) / a
    out["p_body"] = (c - o) / rng
    return out[PORTABLE_ORDER].clip(-50, 50)
