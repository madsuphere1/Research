"""Session & time features (MARKET_RULES.md §11). All clock features are
cyclical-encoded; session flags use UTC wall time (fx convention)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)
    hour = idx.hour + idx.minute / 60

    out["ses_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["ses_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["ses_dow"] = idx.dayofweek.astype(float)
    out["ses_month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["ses_month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    # session flags (UTC): Asia 00-07, London 07-12 (kill zone 07-10),
    # NY 12-21 (kill zone 12-15), overlap 12-16
    out["ses_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
    out["ses_london"] = ((hour >= 7) & (hour < 12)).astype(float)
    out["ses_london_kz"] = ((hour >= 7) & (hour < 10)).astype(float)
    out["ses_ny"] = ((hour >= 12) & (hour < 21)).astype(float)
    out["ses_ny_kz"] = ((hour >= 12) & (hour < 15)).astype(float)
    out["ses_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)

    # Asian range: width and current position vs that range (intraday only)
    daily = idx.normalize()
    is_asia = out["ses_asia"] > 0
    asia_hi = df["high"].where(is_asia).groupby(daily).cummax().ffill()
    asia_lo = df["low"].where(is_asia).groupby(daily).cummin().ffill()
    width = (asia_hi - asia_lo).replace(0, np.nan)
    out["ses_asia_range_w"] = width / df["close"]
    out["ses_asia_range_pos"] = ((df["close"] - asia_lo) / width).clip(-2, 3)

    # bars are intraday? if daily bars, session flags are meaningless -> NaN them
    med = idx.to_series().diff().median()
    if pd.notna(med) and med >= pd.Timedelta(hours=23):
        for col in ["ses_hour_sin", "ses_hour_cos", "ses_asia", "ses_london", "ses_london_kz",
                    "ses_ny", "ses_ny_kz", "ses_overlap", "ses_asia_range_w", "ses_asia_range_pos"]:
            out[col] = np.nan
    return out
