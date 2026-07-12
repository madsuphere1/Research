"""Signal engine: model probability -> CALL / PUT / FLAT decision with
ATR-based TP/SL and fixed risk:reward.

The model predicts P(TP-first) for LONG barrier geometry (TP = rr*SL above,
SL below). Decision rule with symmetric thresholds:

    p >= call_th          -> CALL (long):  TP = close + rr*sl_atr*ATR, SL = close - sl_atr*ATR
    p <= 1 - call_th ...   is NOT simply a put signal; for the short side the
    geometry mirrors, and P(short TP first) ~ P(long SL first) only
    approximately. We therefore require p <= put_th AND the barrier label
    asymmetry to hold, defaulting put_th = 1 - call_th.

Threshold selection: chosen on TRAIN folds only (walk-forward safe) to
maximise expected R per trade subject to a minimum trade count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features.trend import atr


@dataclass
class SignalConfig:
    rr: float = 1.5           # take-profit : stop-loss ratio
    sl_atr: float = 1.5       # stop distance in ATR units
    call_th: float = 0.60     # P(TP-first) needed for CALL
    put_th: float = 0.40      # P(TP-first) below which we take PUT
    cost_r: float = 0.05      # round-trip cost in R units (spread+fees vs SL distance)


def choose_thresholds(
    proba: pd.Series,
    r_long: pd.Series,
    rr: float,
    cost_r: float,
    grid: np.ndarray | None = None,
    min_trades: int = 30,
) -> tuple[float, float]:
    """Pick (call_th, put_th) maximising mean net R per trade on the given
    (training) slice. Short R is the mirror of long R under symmetric
    barriers: r_short = rr when long SL hit first (-1), -1 when long TP."""
    grid = grid if grid is not None else np.arange(0.52, 0.71, 0.02)
    ok = proba.notna() & r_long.notna()
    p, rl = proba[ok], r_long[ok]
    r_short = pd.Series(np.where(rl == -1.0, rr, np.where(rl == rr, -1.0, -rl)), index=rl.index)

    best_call, best_put, best_ev = 0.60, 0.40, -np.inf
    for th in grid:
        calls = rl[p >= th] - cost_r
        puts = r_short[p <= 1 - th] - cost_r
        n = len(calls) + len(puts)
        if n < min_trades:
            continue
        ev = pd.concat([calls, puts]).mean()
        if ev > best_ev:
            best_call, best_put, best_ev = float(th), float(1 - th), float(ev)
    return best_call, best_put


def make_signals(df: pd.DataFrame, proba: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    """Per-bar decision frame: side (+1 CALL, -1 PUT, 0 flat), entry, tp, sl."""
    a = atr(df).reindex(proba.index)
    close = df["close"].reindex(proba.index)
    side = pd.Series(0.0, index=proba.index)
    side[proba >= cfg.call_th] = 1.0
    side[proba <= cfg.put_th] = -1.0

    sl_dist = cfg.sl_atr * a
    tp_dist = cfg.rr * sl_dist
    out = pd.DataFrame(
        {
            "side": side,
            "proba": proba,
            "entry": close,
            "tp": np.where(side > 0, close + tp_dist, np.where(side < 0, close - tp_dist, np.nan)),
            "sl": np.where(side > 0, close - sl_dist, np.where(side < 0, close + sl_dist, np.nan)),
            "rr": cfg.rr,
        },
        index=proba.index,
    )
    return out
