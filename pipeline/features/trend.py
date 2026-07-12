"""Trend typing: strength, regime, expansion/compression, exhaustion,
parabolic and mean-reversion tendencies (MARKET_RULES.md §2)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df.high - df.low,
            (df.high - df.close.shift()).abs(),
            (df.low - df.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff()
    dn = -df.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman ER: |net move| / sum(|bar moves|). 1 = perfect trend, 0 = chop."""
    change = close.diff(n).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    a14 = atr(df)

    out["trend_ema_align"] = np.sign(e20 - e50) + np.sign(e50 - e200)  # -2..2
    out["trend_slope20"] = e20.diff(5) / a14
    out["trend_slope50"] = e50.diff(10) / a14
    out["trend_adx"] = adx(df)
    out["trend_efficiency"] = efficiency_ratio(close)

    # regime code: 2 strong up, 1 weak up, 0 sideways, -1 weak down, -2 strong down
    direction = np.sign(out["trend_slope50"]).fillna(0)
    strong = (out["trend_adx"] > 25) & (out["trend_efficiency"] > 0.3)
    out["trend_regime"] = direction * np.where(strong, 2, 1)
    out.loc[out["trend_adx"] < 18, "trend_regime"] = 0

    # expansion / compression: ATR now vs ATR 20 bars ago
    out["trend_expansion"] = (a14 / a14.shift(20)).clip(0, 5)

    # acceleration: slope of the slope
    out["trend_accel"] = out["trend_slope20"].diff(5)

    # parabolic: price stretched far above/below its own EMA in ATR units
    out["trend_parabolic"] = ((close - e50) / a14).clip(-10, 10)

    # exhaustion: strong trend + falling efficiency
    out["trend_exhaustion"] = (
        (out["trend_adx"] > 30) & (out["trend_efficiency"].diff(10) < -0.1)
    ).astype(float) * direction

    # mean-reversion tendency: 1-lag autocorrelation of returns (rolling)
    r = close.pct_change()
    out["trend_meanrev"] = r.rolling(50).corr(r.shift(1))
    return out
