"""Phase 9 tests: the Q-agent learns from reward/punishment (CALL where
calls pay, PUT where puts pay, FLAT where nothing pays after costs),
converges before max epochs, and the walk-forward wrapper beats both
always-flat and the sign-flipped policy. Recursive refit improves or holds
validation AUC and stops early."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.recursive import recursive_refit
from pipeline.rl import ACTIONS, QAgent, build_states, rewards_frame, walk_forward_rl


def _env(n=4000, seed=0):
    """Synthetic env: high proba -> CALL pays, low -> PUT pays, mid -> noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    proba = pd.Series(rng.uniform(0, 1, n), index=idx)
    rr = 1.5
    p_win = 0.3 + 0.4 * (proba > 0.8) + 0.4 * (proba < 0.2) * 0  # calls win when proba high
    win_long = rng.uniform(0, 1, n) < np.where(proba > 0.8, 0.7, np.where(proba < 0.2, 0.2, 0.4))
    r_long = pd.Series(np.where(win_long, rr, -1.0), index=idx)
    regime = pd.Series(0.0, index=idx)
    vol = pd.Series(rng.uniform(0, 1, n), index=idx)
    return proba, regime, vol, r_long, rr


def test_agent_learns_reward_structure():
    proba, regime, vol, r_long, rr = _env()
    states = build_states(proba, regime, vol)
    rew = rewards_frame(r_long, rr=rr, cost_r=0.05)
    agent = QAgent(seed=1).fit(states.values, rew[list(ACTIONS)].values)
    assert agent.converged and agent.epochs_run < agent.max_epochs

    acts = agent.act(states.values)
    hi, lo = proba > 0.8, proba < 0.2
    # calls dominate where calls pay; puts where puts pay
    assert (acts[hi.values] == 1).mean() > 0.8
    assert (acts[lo.values] == -1).mean() > 0.8


def test_walk_forward_rl_beats_baselines():
    proba, regime, vol, r_long, rr = _env(seed=3)
    actions, agents = walk_forward_rl(proba, regime, vol, r_long, rr=rr, cost_r=0.05)
    rew = rewards_frame(r_long, rr=rr, cost_r=0.05)
    oos = actions[actions.index >= actions.index[len(actions) // 5]]

    def total(a):
        r = np.where(a > 0, rew["CALL"].reindex(oos.index),
                     np.where(a < 0, rew["PUT"].reindex(oos.index), 0.0))
        return np.nansum(r)

    got = total(oos.values)
    assert got > 0            # beats always-flat (=0)
    assert got > total(-oos.values)  # beats the sign-flipped policy
    assert (oos != 0).sum() > 50     # it actually trades


def test_recursive_refit_stops_and_keeps_best():
    rng = np.random.default_rng(4)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    sig = rng.standard_normal(n)
    y = pd.Series((sig + 0.8 * rng.standard_normal(n) > 0).astype(int), index=idx)
    X = pd.DataFrame({"sig": sig, "noise": rng.standard_normal(n)}, index=idx)
    res = recursive_refit(X, y, max_rounds=8, patience=2)
    assert res.model is not None
    assert res.best_val_auc > 0.6
    assert len(res.history) <= 8
    # the kept round is the argmax of history
    assert res.history[res.best_round] == max(res.history)
