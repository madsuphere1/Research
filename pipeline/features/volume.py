"""Volume behaviour + simple rolling volume profile (MARKET_RULES.md §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, profile_window: int = 120, profile_bins: int = 24) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    if v.fillna(0).eq(0).all():  # instruments without volume (some forex feeds)
        for col in ["vol_z", "vol_climax", "vol_dryup", "vol_trend", "vol_price_div",
                    "vol_obv_slope", "vol_ad_slope", "vol_poc_dist", "vol_va_pos"]:
            out[col] = np.nan
        return out

    mu = v.rolling(50).mean()
    sd = v.rolling(50).std().replace(0, np.nan)
    out["vol_z"] = ((v - mu) / sd).clip(-5, 10)
    out["vol_climax"] = (out["vol_z"] > 3).astype(float)
    out["vol_dryup"] = (v < 0.4 * mu).astype(float)
    out["vol_trend"] = (mu / mu.shift(20)).clip(0, 5)  # participation up/down

    # divergence: price pushing on but volume fading (rolling correlation)
    out["vol_price_div"] = c.pct_change().abs().rolling(20).corr(v.pct_change())

    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["vol_obv_slope"] = obv.diff(20) / v.rolling(20).sum().replace(0, np.nan)

    rng = (df.high - df.low).replace(0, np.nan)
    clv = ((c - df.low) - (df.high - c)) / rng  # accumulation/distribution
    ad = (clv * v).cumsum()
    out["vol_ad_slope"] = ad.diff(20) / v.rolling(20).sum().replace(0, np.nan)

    # rolling volume profile: POC distance + position in value area
    # (concept from external/py-market-profile, computed causally per bar)
    poc = np.full(len(df), np.nan)
    va_pos = np.full(len(df), np.nan)
    cv, vv = c.values, v.fillna(0).values
    for i in range(profile_window, len(df)):
        seg_c = cv[i - profile_window : i]
        seg_v = vv[i - profile_window : i]
        hist, edges = np.histogram(seg_c, bins=profile_bins, weights=seg_v)
        if hist.sum() == 0:
            continue
        poc_px = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
        poc[i] = poc_px
        order = np.argsort(hist)[::-1]
        cum, sel = 0.0, np.zeros(profile_bins, bool)
        for b in order:
            sel[b] = True
            cum += hist[b]
            if cum >= 0.7 * hist.sum():
                break
        va_lo = edges[np.argmax(sel)]
        va_hi = edges[profile_bins - np.argmax(sel[::-1])]
        va_pos[i] = (cv[i] - va_lo) / (va_hi - va_lo) if va_hi > va_lo else np.nan
    out["vol_poc_dist"] = (c - poc) / c
    out["vol_va_pos"] = pd.Series(va_pos, index=df.index).clip(-1, 2)
    return out
