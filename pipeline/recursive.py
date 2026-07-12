"""Recursive error-boosted refit (Phase 9).

"Recursively learn until the prediction is true" — done the only honest
way: the model retrains repeatedly, each round up-weighting the training
samples it got wrong (negative feedback), and keeps the round with the
best score on a VALIDATION slice it never trained on. Recursion stops when
validation stops improving (patience) — not when training error hits zero,
because a model can always memorise its training data ("prediction is
true" in-sample) while getting worse on the future. The final quality
claim still comes from walk-forward evaluation downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import LGB_PARAMS


@dataclass
class RecursiveResult:
    model: object
    best_round: int
    history: list = field(default_factory=list)  # val AUC per round

    @property
    def best_val_auc(self) -> float:
        return max(self.history) if self.history else np.nan


def recursive_refit(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.25,
    purge: int = 24,
    boost: float = 0.5,
    max_rounds: int = 8,
    patience: int = 2,
    seed: int = 0,
) -> RecursiveResult:
    """Temporal split: fit on the early part, validate on the late part.
    Each round multiplies weights of misclassified fit-samples by
    exp(boost) (capped), refits, and keeps the best-validating round."""
    n = len(X)
    cut = int(n * (1 - val_frac))
    Xf, yf = X.iloc[: cut - purge], y.iloc[: cut - purge]
    Xv, yv = X.iloc[cut:], y.iloc[cut:]

    w = np.ones(len(Xf))
    best_auc, best_model, best_round, since_best = -np.inf, None, 0, 0
    history = []
    for rnd in range(max_rounds):
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed + rnd})
        m.fit(Xf, yf, sample_weight=w)
        val_auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1]) if yv.nunique() > 1 else np.nan
        history.append(float(val_auc))
        if val_auc > best_auc:
            best_auc, best_model, best_round, since_best = val_auc, m, rnd, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
        # negative feedback: samples the model still gets wrong weigh more
        wrong = (m.predict(Xf) != yf.values)
        w = np.clip(w * np.exp(boost * wrong), None, 20.0)
        w *= len(w) / w.sum()
    return RecursiveResult(model=best_model, best_round=best_round, history=history)
