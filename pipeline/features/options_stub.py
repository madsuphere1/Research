"""Options-chain feature schema (MARKET_RULES.md §12).

No free keyless options-chain source is reachable from this environment, so
these stay NaN and the relevance stage drops them. Plug a chain provider in
by implementing `compute` for real; keep the column names stable so the
model/MQL5 stages need no changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OPTION_COLS = [
    "opt_iv", "opt_iv_rank", "opt_iv_pctile",
    "opt_delta", "opt_gamma", "opt_theta", "opt_vega",
    "opt_oi", "opt_oi_change", "opt_pcr", "opt_max_pain_dist",
    "opt_gex", "opt_dte",
]


def compute(df: pd.DataFrame, chain: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=df.index, columns=OPTION_COLS)
    if chain is not None:
        aligned = chain.reindex(df.index).ffill()
        for col in OPTION_COLS:
            if col in aligned:
                out[col] = aligned[col]
    return out
