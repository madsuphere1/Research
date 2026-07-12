"""Cost-aware event backtest over the signal frame.

One position at a time; entry at next bar open after the signal (no
same-bar fill fantasy); exit at TP/SL touch (conservative: if both touch
within one bar, SL is assumed first) or at the horizon. Costs deducted
per round trip in R units. Reports net expectancy, win rate, profit
factor, max drawdown in R, and a permutation p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    net_r_per_trade: float
    win_rate: float
    profit_factor: float
    max_dd_r: float
    n_trades: int
    p_value: float
    equity_r: pd.Series = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "net_r_per_trade": round(self.net_r_per_trade, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "max_dd_r": round(self.max_dd_r, 2),
            "p_value": round(self.p_value, 4),
        }


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int = 24,
    cost_r: float = 0.05,
    n_perm: int = 500,
    seed: int = 0,
) -> BacktestResult:
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    pos = {ts: i for i, ts in enumerate(df.index)}
    recs = []
    busy_until = -1
    for ts, row in signals[signals.side != 0].iterrows():
        i = pos[ts]
        if i <= busy_until or i + 1 >= len(df):
            continue
        entry_i = i + 1
        entry = o[entry_i]  # fill at next bar open
        sl_dist = abs(row.entry - row.sl)
        if sl_dist <= 0 or np.isnan(sl_dist):
            continue
        side = row.side
        tp = entry + side * row.rr * sl_dist
        sl = entry - side * sl_dist
        end = min(entry_i + horizon, len(df) - 1)
        r, exit_i = None, end
        for j in range(entry_i, end + 1):
            hit_tp = h[j] >= tp if side > 0 else l[j] <= tp
            hit_sl = l[j] <= sl if side > 0 else h[j] >= sl
            if hit_sl:  # conservative: SL wins ties
                r, exit_i = -1.0, j
                break
            if hit_tp:
                r, exit_i = row.rr, j
                break
        if r is None:
            r = side * (c[end] - entry) / sl_dist
        recs.append({"time": ts, "side": side, "entry": entry, "tp": tp, "sl": sl,
                     "exit_i": exit_i, "r_gross": r, "r_net": r - cost_r,
                     "proba": row.proba})
        busy_until = exit_i

    trades = pd.DataFrame(recs)
    if trades.empty:
        return BacktestResult(trades, 0.0, 0.0, 0.0, 0.0, 0, 1.0, pd.Series(dtype=float))

    rn = trades.r_net
    equity = rn.cumsum()
    dd = (equity.cummax() - equity).max()
    wins, losses = rn[rn > 0], rn[rn <= 0]
    pf = wins.sum() / max(1e-9, -losses.sum())

    # sign-flip permutation test: if direction choice carried no information,
    # how often would random signs beat the observed mean net R?
    rng = np.random.default_rng(seed)
    obs = rn.mean()
    beats = 0
    for _ in range(n_perm):
        signs = rng.choice([1.0, -1.0], size=len(rn))
        beats += (rn.values * signs).mean() >= obs
    p_val = (beats + 1) / (n_perm + 1)

    return BacktestResult(
        trades=trades,
        net_r_per_trade=float(obs),
        win_rate=float((rn > 0).mean()),
        profit_factor=float(pf),
        max_dd_r=float(dd),
        n_trades=len(trades),
        p_value=float(p_val),
        equity_r=equity,
    )
