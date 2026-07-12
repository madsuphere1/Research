"""Reinforcement-learning signal agent (Phase 9).

A contextual Q-learner sits on top of the supervised model: its state is
(model-probability bin, HMM/trend regime bin, volatility bin); its actions
are FLAT / CALL / PUT; its reward is the realised net R of the trade the
action would open (negative R = punishment). Each bar is an episode
(trades don't chain), so Q-learning reduces to a contextual bandit — the
honest formulation for barrier-style trades.

"Recursive learning until true": the agent replays the TRAINING segment
for up to `max_epochs`, updating Q from reward feedback with decaying
exploration, and stops when the policy is stable (max |dQ| < tol) — i.e.
until further recursion changes nothing. It never sees evaluation data
during learning; walk-forward evaluation is the only claim of quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACTIONS = ("FLAT", "CALL", "PUT")
N_PROBA_BINS = 5
N_REGIME_BINS = 3
N_VOL_BINS = 3


def build_states(proba: pd.Series, regime: pd.Series, vol: pd.Series) -> pd.Series:
    """Discretise (proba, regime, vol) into a single integer state id."""
    p = np.clip((proba.values * N_PROBA_BINS).astype(int), 0, N_PROBA_BINS - 1)
    r = pd.cut(regime, bins=[-np.inf, -0.5, 0.5, np.inf], labels=False).fillna(1).astype(int).values
    v_ranks = vol.rank(pct=True).fillna(0.5).values
    v = np.clip((v_ranks * N_VOL_BINS).astype(int), 0, N_VOL_BINS - 1)
    state = p * (N_REGIME_BINS * N_VOL_BINS) + r * N_VOL_BINS + v
    return pd.Series(state, index=proba.index)


def rewards_frame(r_long: pd.Series, rr: float, cost_r: float) -> pd.DataFrame:
    """Reward per action per bar. Short mirrors long under symmetric barriers."""
    r_short = pd.Series(
        np.where(r_long == -1.0, rr, np.where(r_long == rr, -1.0, -r_long)),
        index=r_long.index,
    )
    return pd.DataFrame(
        {"FLAT": 0.0, "CALL": r_long - cost_r, "PUT": r_short - cost_r},
        index=r_long.index,
    )


@dataclass
class QAgent:
    n_states: int = N_PROBA_BINS * N_REGIME_BINS * N_VOL_BINS
    lr: float = 0.05
    eps0: float = 0.3
    tol: float = 1e-3
    max_epochs: int = 200
    seed: int = 0
    Q: np.ndarray = field(default=None, repr=False)
    epochs_run: int = 0
    converged: bool = False

    def fit(self, states: np.ndarray, rewards: np.ndarray) -> "QAgent":
        """Replay the training segment until the value table stops moving.

        rewards: (n, 3) array aligned with states; NaN rows are skipped.
        """
        rng = np.random.default_rng(self.seed)
        self.Q = np.zeros((self.n_states, len(ACTIONS)))
        counts = np.zeros_like(self.Q)
        ok = ~np.isnan(rewards).any(axis=1)
        s_ok, r_ok = states[ok], rewards[ok]
        for epoch in range(self.max_epochs):
            eps = self.eps0 * (1 - epoch / self.max_epochs)
            q_before = self.Q.copy()
            for s, rew in zip(s_ok, r_ok):
                if rng.random() < eps:
                    a = rng.integers(0, len(ACTIONS))       # explore
                else:
                    a = int(np.argmax(self.Q[s]))            # exploit
                # feedback: positive reward reinforces, negative punishes.
                # sample-average step size (1/N) so Q provably converges to
                # the true expected reward despite stochastic outcomes
                counts[s, a] += 1
                self.Q[s, a] += (rew[a] - self.Q[s, a]) / counts[s, a]
            self.epochs_run = epoch + 1
            if np.abs(self.Q - q_before).max() < self.tol:
                self.converged = True
                break
        # states never explored keep Q=0 -> policy falls back to FLAT
        return self

    def act(self, states: np.ndarray) -> np.ndarray:
        """Greedy policy: 0 FLAT, +1 CALL, -1 PUT."""
        a = self.Q[states].argmax(axis=1)
        return np.where(a == 1, 1.0, np.where(a == 2, -1.0, 0.0))


def walk_forward_rl(
    proba: pd.Series,
    regime: pd.Series,
    vol: pd.Series,
    r_long: pd.Series,
    rr: float,
    cost_r: float,
    n_folds: int = 4,
    purge: int = 24,
    seed: int = 0,
) -> tuple[pd.Series, list[QAgent]]:
    """Train on past folds, act greedily on each next fold. Returns the
    out-of-sample action series (+1 CALL / -1 PUT / 0 FLAT)."""
    ok = proba.notna()
    states = build_states(proba[ok], regime.reindex(proba.index)[ok],
                          vol.reindex(proba.index)[ok])
    rew = rewards_frame(r_long.reindex(states.index), rr, cost_r)
    sv, rv = states.values, rew[list(ACTIONS)].values

    n = len(sv)
    edges = np.linspace(n // (n_folds + 1), n, n_folds + 1, dtype=int)
    actions = pd.Series(0.0, index=states.index)
    agents = []
    for f in range(n_folds):
        tr_end, te_lo, te_hi = edges[f] - purge, edges[f], edges[f + 1]
        if tr_end < 200:
            continue
        agent = QAgent(seed=seed + f).fit(sv[:tr_end], rv[:tr_end])
        actions.iloc[te_lo:te_hi] = agent.act(sv[te_lo:te_hi])
        agents.append(agent)
    return actions, agents
