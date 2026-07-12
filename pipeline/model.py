"""Purged walk-forward LightGBM + per-instrument feature relevance.

Lessons enforced from ../Claude-researcg: evaluation is only ever
out-of-sample walk-forward, with a purge gap >= label horizon so training
labels can't overlap test features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class WalkForwardResult:
    proba: pd.Series                    # P(TP-first | long geometry) out-of-sample
    auc_by_fold: list = field(default_factory=list)
    feature_importance: pd.Series | None = None
    kept_features: list = field(default_factory=list)
    dropped_features: list = field(default_factory=list)
    models: list = field(default_factory=list)

    @property
    def auc(self) -> float:
        return float(np.mean(self.auc_by_fold)) if self.auc_by_fold else np.nan


LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_estimators=300,
    verbosity=-1,
)


def select_relevant_features(
    X: pd.DataFrame, y: pd.Series, n_shadow: int = 5, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Per-instrument relevance: keep features whose gain importance beats the
    best of `n_shadow` shuffled 'shadow' copies (a light Boruta). Features
    that are entirely NaN (e.g. options without a chain, sessions on daily
    bars) are dropped first."""
    dead = [c for c in X.columns if X[c].notna().sum() < len(X) * 0.2 or X[c].nunique() <= 1]
    Xv = X.drop(columns=dead)
    rng = np.random.default_rng(seed)
    shadows = {}
    for i in range(n_shadow):
        col = Xv.iloc[:, rng.integers(0, Xv.shape[1])].sample(frac=1, random_state=i).values
        shadows[f"__shadow_{i}"] = col
    Xs = pd.concat([Xv, pd.DataFrame(shadows, index=Xv.index)], axis=1)
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 200, "random_state": seed})
    m.fit(Xs, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=Xs.columns)
    threshold = imp[[c for c in Xs.columns if c.startswith("__shadow_")]].max()
    kept = [c for c in Xv.columns if imp[c] > threshold]
    dropped = dead + [c for c in Xv.columns if imp[c] <= threshold]
    if len(kept) < 10:  # degenerate case: fall back to top-20 by gain
        kept = imp.drop(index=[c for c in imp.index if c.startswith("__shadow_")]).nlargest(20).index.tolist()
        dropped = [c for c in X.columns if c not in kept]
    return kept, dropped


def walk_forward(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    n_folds: int = 5,
    purge: int = 24,
    min_train: int = 500,
    seed: int = 0,
) -> WalkForwardResult:
    """Expanding-window walk-forward. Binary target: TP-first (label==1)
    vs not. Purge `purge` bars (>= label horizon) between train and test."""
    y = (labels["label"] == 1).astype(int)
    ok = labels["label"].notna()
    X, y = X[ok], y[ok]

    kept, dropped = select_relevant_features(X.iloc[: max(min_train, len(X) // 3)],
                                             y.iloc[: max(min_train, len(X) // 3)], seed=seed)
    Xk = X[kept]

    n = len(Xk)
    fold_edges = np.linspace(max(min_train, n // (n_folds + 1)), n, n_folds + 1, dtype=int)
    proba = pd.Series(np.nan, index=Xk.index)
    res = WalkForwardResult(proba=proba, kept_features=kept, dropped_features=dropped)
    importances = []

    for f in range(n_folds):
        tr_end = fold_edges[f] - purge
        te_lo, te_hi = fold_edges[f], fold_edges[f + 1]
        if tr_end < min_train // 2 or te_hi - te_lo < 30:
            continue
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed + f})
        m.fit(Xk.iloc[:tr_end], y.iloc[:tr_end])
        p = m.predict_proba(Xk.iloc[te_lo:te_hi])[:, 1]
        proba.iloc[te_lo:te_hi] = p
        if y.iloc[te_lo:te_hi].nunique() > 1:
            res.auc_by_fold.append(roc_auc_score(y.iloc[te_lo:te_hi], p))
        importances.append(pd.Series(m.booster_.feature_importance("gain"), index=kept))
        res.models.append(m)

    if importances:
        res.feature_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    res.proba = proba
    return res


def fit_final_model(X: pd.DataFrame, labels: pd.DataFrame, kept: list[str], seed: int = 0):
    """Fit on all labelled data (deployment model exported to MQL5 params)."""
    y = (labels["label"] == 1).astype(int)
    ok = labels["label"].notna()
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
    m.fit(X[ok][kept], y[ok])
    return m
