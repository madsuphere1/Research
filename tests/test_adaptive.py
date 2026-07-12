"""Tests for live correction: the circuit breaker halts a losing streak
(and saves money vs no breaker), and in-window retraining relearns a
relationship that inverts mid-window while a static model stays wrong."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb

from pipeline.labeling import triple_barrier
from pipeline.model import LGB_PARAMS
from pipeline.signal import SignalConfig
from pipeline.signal_actions import signals_from_actions
from pipeline.simulate import adaptive_proba, simulate_dollars


def _downtrend(n=600, price=100.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = price * np.exp(np.cumsum(np.full(n, -0.002)))
    return pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


def test_circuit_breaker_stops_losing_streak():
    df = _downtrend()
    actions = pd.Series(0.0, index=df.index)
    actions.iloc[10:500:10] = 1.0  # keep buying a falling market -> every trade loses
    cfg = SignalConfig(rr=1.5, sl_atr=1.5)
    sig = signals_from_actions(df, actions, cfg)

    s_off, t_off = simulate_dollars(df, sig, 10_000, 1.0, 2.0, horizon=24,
                                    cost_r=0.0, breaker_lookback=0)
    s_on, t_on = simulate_dollars(df, sig, 10_000, 1.0, 2.0, horizon=24,
                                  cost_r=0.0, breaker_lookback=5,
                                  breaker_stop_r=-3.0, cooldown_bars=100)
    assert s_off["breaker_trips"] == 0
    assert s_on["breaker_trips"] >= 1
    assert s_on["signals_skipped_while_paused"] > 0
    assert s_on["n_trades"] < s_off["n_trades"]          # it actually stood down
    assert s_on["final_balance"] > s_off["final_balance"]  # and lost less


def test_in_window_retraining_relearns_inverted_signal():
    """Feature 'sig' predicts direction; the relationship INVERTS at the
    window start. A static pre-window model stays wrong; adaptive_proba
    relearns within a few retrain cycles."""
    rng = np.random.default_rng(7)
    n, flip = 4000, 3000
    sig = rng.standard_normal(n)
    coef = np.where(np.arange(n) < flip, 1.0, -1.0)  # inversion at sim_start
    ret = 0.004 * coef * np.roll(sig, 1) + 0.001 * rng.standard_normal(n)
    ret[0] = 0
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.roll(close, 1), "high": close * 1.004,
                       "low": close * 0.996, "close": close, "volume": 1.0},
                      index=idx)
    df.iloc[0, 0] = close[0]
    X = pd.DataFrame({"sig": sig, "noise": rng.standard_normal(n)}, index=idx)
    lab = triple_barrier(df, horizon=8, sl_atr=1.0, rr=1.0)
    sim_start = idx[flip]

    # static: fit once on pre-window data only
    mask = (lab["label"].notna()) & (np.arange(n) < flip - 8)
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": 0})
    m.fit(X[mask], (lab["label"] == 1).astype(int)[mask])
    static = pd.Series(m.predict_proba(X)[:, 1], index=idx)[idx >= sim_start]

    adaptive, n_retrains = adaptive_proba(X, lab, ["sig", "noise"], sim_start,
                                          purge=8, retrain_every=100,
                                          train_window=600)
    assert n_retrains >= 8

    # score both on the LATE part of the window (after retraining had data)
    late = idx[(np.arange(n) >= flip + 500) & (np.arange(n) < n - 8)]
    y_late = (lab["label"] == 1).astype(int).loc[late]
    from sklearn.metrics import roc_auc_score
    auc_static = roc_auc_score(y_late, static.loc[late])
    auc_adaptive = roc_auc_score(y_late, adaptive.loc[late])
    assert auc_static < 0.4, f"static should be inverted/wrong, got {auc_static}"
    assert auc_adaptive > 0.6, f"adaptive should have relearned, got {auc_adaptive}"
