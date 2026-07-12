"""Signal frame from an explicit action series (RL policy output)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features.trend import atr
from .signal import SignalConfig


def signals_from_actions(df: pd.DataFrame, actions: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    a = atr(df).reindex(actions.index)
    close = df["close"].reindex(actions.index)
    side = actions.astype(float)
    sl_dist = cfg.sl_atr * a
    tp_dist = cfg.rr * sl_dist
    return pd.DataFrame(
        {
            "side": side,
            "proba": np.nan,
            "entry": close,
            "tp": np.where(side > 0, close + tp_dist, np.where(side < 0, close - tp_dist, np.nan)),
            "sl": np.where(side > 0, close - sl_dist, np.where(side < 0, close + sl_dist, np.nan)),
            "rr": cfg.rr,
        },
        index=actions.index,
    )
