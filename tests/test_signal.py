"""Phase 4 tests: signal geometry (TP/SL/RR), threshold picker, backtest
mechanics (next-bar entry, conservative tie handling, costs, one position at
a time), and that an informed signal beats a random one."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.backtest import run_backtest
from pipeline.labeling import triple_barrier
from pipeline.signal import SignalConfig, choose_thresholds, make_signals


def synth(n=2000, seed=11):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n)
    ret = 0.004 * np.roll(sig, 1) + 0.004 * rng.standard_normal(n)
    ret[0] = 0
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.roll(close, 1), "high": close * 1.003,
                       "low": close * 0.997, "close": close, "volume": 1.0}, index=idx)
    df.iloc[0, 0] = close[0]
    return df, pd.Series(sig, index=idx)


def test_signal_geometry():
    df, sig = synth()
    proba = pd.Series(0.5, index=df.index)
    proba.iloc[100] = 0.9   # CALL
    proba.iloc[200] = 0.05  # PUT
    cfg = SignalConfig(rr=2.0, sl_atr=1.5)
    s = make_signals(df, proba, cfg)
    call = s.iloc[100]
    assert call.side == 1 and call.tp > call.entry > call.sl
    assert np.isclose((call.tp - call.entry) / (call.entry - call.sl), 2.0)
    put = s.iloc[200]
    assert put.side == -1 and put.tp < put.entry < put.sl
    assert np.isclose((put.entry - put.tp) / (put.sl - put.entry), 2.0)
    assert (s.side == 0).sum() == len(df) - 2


def test_choose_thresholds_prefers_confident():
    df, sig = synth()
    lab = triple_barrier(df, horizon=12, sl_atr=1.0, rr=1.5)
    # fake calibrated proba: high when future r_long is a win
    proba = pd.Series(0.5 + 0.3 * (lab.r_long == 1.5) - 0.3 * (lab.r_long == -1.0),
                      index=df.index)
    call_th, put_th = choose_thresholds(proba, lab.r_long, rr=1.5, cost_r=0.05)
    assert call_th >= 0.52 and np.isclose(call_th + put_th, 1.0)


def test_backtest_mechanics_and_informed_beats_random():
    df, sig = synth()
    lab = triple_barrier(df, horizon=12, sl_atr=1.5, rr=1.5)
    # informed proba built from the planted signal (known at bar t, predicts t+1)
    proba_inf = pd.Series(1 / (1 + np.exp(-2.0 * sig.values)), index=df.index)
    cfg = SignalConfig(rr=1.5, sl_atr=1.5, call_th=0.75, put_th=0.25)
    bt_inf = run_backtest(df, make_signals(df, proba_inf, cfg), horizon=12, cost_r=0.02)

    rng = np.random.default_rng(0)
    proba_rnd = pd.Series(rng.uniform(0, 1, len(df)), index=df.index)
    bt_rnd = run_backtest(df, make_signals(df, proba_rnd, cfg), horizon=12, cost_r=0.02)

    assert bt_inf.n_trades > 30
    assert bt_inf.net_r_per_trade > bt_rnd.net_r_per_trade
    assert bt_inf.net_r_per_trade > 0
    assert bt_inf.p_value < 0.05
    # one position at a time: exits never overlap next entry
    t = bt_inf.trades
    pos = {ts: i for i, ts in enumerate(df.index)}
    entry_pos = t.time.map(pos).values + 1
    assert (entry_pos[1:] > t.exit_i.values[:-1]).all()


def test_backtest_costs_reduce_expectancy():
    df, sig = synth()
    proba = pd.Series(1 / (1 + np.exp(-2.0 * sig.values)), index=df.index)
    cfg = SignalConfig(call_th=0.7, put_th=0.3)
    cheap = run_backtest(df, make_signals(df, proba, cfg), cost_r=0.0)
    dear = run_backtest(df, make_signals(df, proba, cfg), cost_r=0.2)
    assert np.isclose(cheap.net_r_per_trade - dear.net_r_per_trade, 0.2, atol=1e-9)
