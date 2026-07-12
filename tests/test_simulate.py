"""Tests for the account simulator: leverage caps position size, balance
compounds per trade, and a losing streak cannot take balance below zero
without being flagged."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.signal import SignalConfig
from pipeline.signal_actions import signals_from_actions
from pipeline.simulate import simulate_dollars


def _df(n=300, price=100.0, updrift=0.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = price * np.exp(np.cumsum(np.full(n, updrift)))
    return pd.DataFrame({"open": close, "high": close * 1.02,
                         "low": close * 0.98, "close": close,
                         "volume": 1.0}, index=idx)


def test_leverage_caps_notional():
    df = _df(updrift=0.001)  # steady rise: every long hits TP
    actions = pd.Series(0.0, index=df.index)
    actions.iloc[50] = 1.0
    cfg = SignalConfig(rr=1.5, sl_atr=1.5)
    sig = signals_from_actions(df, actions, cfg)
    # tiny SL distance vs balance -> uncapped qty would exceed leverage 1
    s1, t1 = simulate_dollars(df, sig, 10_000, leverage=1.0, risk_pct=50.0,
                              horizon=24, cost_r=0.0)
    s2, t2 = simulate_dollars(df, sig, 10_000, leverage=10.0, risk_pct=50.0,
                              horizon=24, cost_r=0.0)
    assert len(t1) == 1 and len(t2) == 1
    assert t1.qty.iloc[0] <= 10_000 / t1.entry.iloc[0] * 1.0001  # capped at 1x
    assert t2.qty.iloc[0] > t1.qty.iloc[0]                        # more leverage, more size
    assert s2["final_balance"] > s1["final_balance"]


def test_balance_compounds_and_survives():
    df = _df(updrift=-0.001)  # steady fall: every long loses
    actions = pd.Series(0.0, index=df.index)
    actions.iloc[10:200:25] = 1.0
    cfg = SignalConfig(rr=1.5, sl_atr=1.5)
    sig = signals_from_actions(df, actions, cfg)
    s, t = simulate_dollars(df, sig, 10_000, leverage=1.0, risk_pct=10.0,
                            horizon=24, cost_r=0.0)
    assert (t.pnl < 0).all()
    # 10% risk per losing trade compounds down, never below zero
    assert 0 < s["final_balance"] < 10_000
    assert not s["busted"]
    losses = -t.pnl.values
    balances = t.balance.values
    # each loss is ~10% of the balance before the trade (leverage may cap lower)
    prev = np.concatenate([[10_000.0], balances[:-1]])
    assert (losses <= prev * 0.101).all()
