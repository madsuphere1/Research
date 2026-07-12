"""Triple-barrier labeling (Lopez de Prado style, cf. external/machine-learning-for-trading).

For each bar t: TP at +rr*sl_atr*ATR_t, SL at -sl_atr*ATR_t (long geometry),
vertical barrier after `horizon` bars. Label:
    +1 TP hit first (a CALL at t would have paid rr:1)
    -1 SL hit first (a PUT would have paid)
     0 vertical barrier first
Also returns the realised R multiple of a long entered at t (used by the
backtest), and first_touch bar offsets (used for purging).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features.trend import atr


def triple_barrier(
    df: pd.DataFrame,
    horizon: int = 24,
    sl_atr: float = 1.5,
    rr: float = 1.5,
) -> pd.DataFrame:
    a = atr(df).values
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(df)
    label = np.zeros(n)
    r_long = np.full(n, np.nan)
    touch = np.full(n, np.nan)

    for i in range(n - 1):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        sl = sl_atr * a[i]
        tp = rr * sl
        up, dn = c[i] + tp, c[i] - sl
        end = min(i + horizon, n - 1)
        lab, rl, tch = 0.0, np.nan, end - i
        for j in range(i + 1, end + 1):
            hit_up, hit_dn = h[j] >= up, l[j] <= dn
            if hit_up and hit_dn:  # both inside one bar -> conservative: SL first
                lab, rl, tch = -1.0, -1.0, j - i
                break
            if hit_up:
                lab, rl, tch = 1.0, rr, j - i
                break
            if hit_dn:
                lab, rl, tch = -1.0, -1.0, j - i
                break
        if np.isnan(rl):  # vertical barrier: mark-to-market R
            rl = (c[end] - c[i]) / sl
        label[i], r_long[i], touch[i] = lab, rl, tch

    out = pd.DataFrame(
        {"label": label, "r_long": r_long, "touch_bars": touch}, index=df.index
    )
    out.iloc[-1] = np.nan
    return out
