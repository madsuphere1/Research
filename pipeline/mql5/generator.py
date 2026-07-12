"""Distill the pipeline model into a portable logistic scorer and render the
MQL5 Expert Advisor.

Why distillation: the LightGBM model consumes ~100 research features (swing
confirmation, volume profile, divergences...) that cannot be re-implemented
bar-identically inside an EA without a huge error surface. Instead we fit a
logistic regression on 12 portable features (exact MQL5 twins in
template.mq5) against the same TP-first labels, measure its own purged
walk-forward AUC honestly, and export it only with its measured quality
printed in the EA header. The full model remains the research reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .portable import PORTABLE_ORDER, compute_portable

TEMPLATE = Path(__file__).with_name("template.mq5")


@dataclass
class DistilledModel:
    coefs: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    intercept: float
    wf_auc: float          # distilled model's own purged walk-forward AUC
    corr_with_full: float  # rank corr of distilled proba vs full-model OOS proba


def distill(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    full_proba: pd.Series | None = None,
    n_folds: int = 4,
    purge: int = 24,
    seed: int = 0,
) -> DistilledModel:
    P = compute_portable(df)
    y = (labels["label"] == 1).astype(int)
    ok = labels["label"].notna() & P.notna().all(axis=1)
    Xp, yb = P[ok].values, y[ok].values

    means, stds = Xp.mean(axis=0), Xp.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (Xp - means) / stds

    # purged walk-forward AUC of the distilled model itself
    n = len(Z)
    edges = np.linspace(n // (n_folds + 1), n, n_folds + 1, dtype=int)
    aucs, oos = [], pd.Series(np.nan, index=P[ok].index)
    for f in range(n_folds):
        tr_end, te_lo, te_hi = edges[f] - purge, edges[f], edges[f + 1]
        if tr_end < 200 or te_hi - te_lo < 50 or len(np.unique(yb[:tr_end])) < 2:
            continue
        m = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
        m.fit(Z[:tr_end], yb[:tr_end])
        p = m.predict_proba(Z[te_lo:te_hi])[:, 1]
        oos.iloc[te_lo:te_hi] = p
        if len(np.unique(yb[te_lo:te_hi])) > 1:
            aucs.append(roc_auc_score(yb[te_lo:te_hi], p))

    final = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
    final.fit(Z, yb)

    corr = np.nan
    if full_proba is not None:
        both = pd.concat([oos.rename("d"), full_proba.rename("f")], axis=1).dropna()
        if len(both) > 50:
            corr = float(both.corr(method="spearman").iloc[0, 1])

    return DistilledModel(
        coefs=final.coef_[0],
        means=means,
        stds=stds,
        intercept=float(final.intercept_[0]),
        wf_auc=float(np.mean(aucs)) if aucs else np.nan,
        corr_with_full=corr,
    )


def _fmt(arr) -> str:
    return ", ".join(f"{v:.10g}" for v in arr)


def render_ea(
    dm: DistilledModel,
    *,
    symbol: str,
    provider: str,
    timeframe: str,
    call_th: float,
    put_th: float,
    sl_atr: float,
    rr: float,
    horizon: int,
    full_auc: float,
    bt_summary: dict,
    risk_pct: float = 1.0,
    magic: int = 20260712,
    ea_name: str | None = None,
) -> str:
    ea_name = ea_name or f"Pipeline_{symbol.replace('-', '').replace('/', '')}_{timeframe}"
    tpl = TEMPLATE.read_text()
    subs = {
        "{EA_NAME}": ea_name,
        "{SYMBOL}": symbol,
        "{PROVIDER}": provider,
        "{TIMEFRAME}": timeframe,
        "{GENERATED_AT}": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "{FULL_AUC}": f"{full_auc:.3f}",
        "{DIST_AUC}": f"{dm.wf_auc:.3f}",
        "{BT_SUMMARY}": str(bt_summary),
        "{RISK_PCT}": f"{risk_pct:g}",
        "{CALL_TH}": f"{call_th:g}",
        "{PUT_TH}": f"{put_th:g}",
        "{SL_ATR}": f"{sl_atr:g}",
        "{RR}": f"{rr:g}",
        "{HORIZON}": str(horizon),
        "{MAGIC}": str(magic),
        "{COEFS}": _fmt(dm.coefs),
        "{MEANS}": _fmt(dm.means),
        "{STDS}": _fmt(dm.stds),
        "{INTERCEPT}": f"{dm.intercept:.10g}",
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    assert "{" + "EA_NAME" + "}" not in tpl
    return tpl


def distill_and_generate(df, labels, full_proba, out_path, **kw) -> DistilledModel:
    dm = distill(df, labels, full_proba)
    code = render_ea(dm, **kw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(code)
    return dm
