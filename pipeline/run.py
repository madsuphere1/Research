"""End-to-end runner.

    python -m pipeline.run --provider coinbase --symbol BTC-USD \
        --timeframe 1h --years 2 --rr 1.5 --sl-atr 1.5 --horizon 24

Stages:
 1. download (cached) and normalise OHLCV
 2. window-permutation research: for every (lookback, offset) window run the
    purged walk-forward and record AUC + which feature families survived
    relevance -> behaviour stability across time
 3. full-window model: relevance -> walk-forward -> OOS probabilities
 4. thresholds chosen on the FIRST HALF of OOS predictions only; the
    cost-aware backtest is scored on the SECOND HALF (untouched)
 5. distill to portable logistic model, generate the MQL5 EA
 6. write outputs/<run>/report.md + trades/predictions parquet + .mq5

Honesty rules: every quoted quality number is out-of-sample and net of
costs; the evaluation half never influences thresholds or model fit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import load_ohlcv, slice_window, window_permutations
from .data.windows import span_months
from .features import build_features
from .labeling import triple_barrier
from .model import fit_final_model, walk_forward
from .mql5.generator import distill, render_ea
from .recursive import recursive_refit
from .rl import walk_forward_rl
from .signal import SignalConfig, choose_thresholds, make_signals
from .signal_actions import signals_from_actions

ROOT = Path(__file__).resolve().parents[1]


def group_of(col: str) -> str:
    return col.split("_")[0]


def run_window_research(df, wins, horizon, sl_atr, rr, purge):
    rows = []
    for w in wins:
        part = slice_window(df, w)
        if len(part) < 1200:
            continue
        X = build_features(part)
        lab = triple_barrier(part, horizon=horizon, sl_atr=sl_atr, rr=rr).loc[X.index]
        res = walk_forward(X, lab, n_folds=4, purge=purge)
        fam = sorted({group_of(c) for c in res.kept_features})
        rows.append({
            "window": w.name, "bars": len(part), "wf_auc": round(res.auc, 4),
            "n_kept": len(res.kept_features), "kept_families": ",".join(fam),
        })
        print(f"  {w.name}: bars={len(part)} AUC={res.auc:.3f} kept={len(res.kept_features)}")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["coinbase", "bitstamp", "kraken", "histdata", "csv"])
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--cost-r", type=float, default=0.05)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--lookbacks", default="6,12,24")
    ap.add_argument("--offsets", default="0,6,12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tag = f"{args.provider}_{args.symbol.replace('/', '').replace('-', '')}_{args.timeframe}"
    out_dir = Path(args.out) if args.out else ROOT / "outputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading {args.provider}:{args.symbol} {args.timeframe} {args.years}y")
    df = load_ohlcv(args.provider, args.symbol, args.timeframe, years=args.years)
    print(f"      {len(df)} bars {df.index.min():%F} .. {df.index.max():%F}")

    print("[2/6] window-permutation research")
    span = span_months(df)
    wins = window_permutations(
        lookbacks=[int(x) for x in args.lookbacks.split(",")],
        offsets=[int(x) for x in args.offsets.split(",")],
        max_span_months=int(np.ceil(span)),
    )
    win_df = run_window_research(df, wins, args.horizon, args.sl_atr, args.rr, args.horizon)
    win_df.to_csv(out_dir / "window_research.csv", index=False)

    print("[3/6] full-window walk-forward model")
    X = build_features(df)
    lab = triple_barrier(df, horizon=args.horizon, sl_atr=args.sl_atr, rr=args.rr).loc[X.index]
    res = walk_forward(X, lab, n_folds=5, purge=args.horizon)
    print(f"      walk-forward AUC {res.auc:.4f} (folds: {[round(a,3) for a in res.auc_by_fold]})")
    print(f"      kept {len(res.kept_features)} features, dropped {len(res.dropped_features)}")

    print("[3b] recursive error-boosted refit (validation-early-stopped)")
    y_bin = (lab["label"] == 1).astype(int)[lab["label"].notna()]
    rec = recursive_refit(X.loc[y_bin.index, res.kept_features], y_bin, purge=args.horizon)
    print(f"      val AUC per round {[round(h, 4) for h in rec.history]} -> kept round {rec.best_round}")

    print("[4/6] thresholds on first half of OOS; backtest on held-out second half")
    oos = res.proba.dropna()
    half = oos.index[len(oos) // 2]
    call_th, put_th = choose_thresholds(
        res.proba[res.proba.index < half], lab["r_long"], rr=args.rr, cost_r=args.cost_r
    )
    cfg = SignalConfig(rr=args.rr, sl_atr=args.sl_atr, call_th=call_th,
                       put_th=put_th, cost_r=args.cost_r)
    eval_proba = res.proba[res.proba.index >= half]
    signals = make_signals(df, eval_proba, cfg)
    bt = run_backtest(df, signals, horizon=args.horizon, cost_r=args.cost_r)
    print(f"      thresholds call>={call_th:.2f} put<={put_th:.2f} | {bt.summary()}")

    print("[4b] RL agent (contextual Q, reward = net R) on the same held-out half")
    regime = X["trend_regime"].reindex(res.proba.index)
    volf = X["vlt_atr_pct"].reindex(res.proba.index)
    rl_actions, rl_agents = walk_forward_rl(
        res.proba, regime, volf, lab["r_long"], rr=args.rr, cost_r=args.cost_r,
        purge=args.horizon,
    )
    rl_eval = rl_actions[rl_actions.index >= half]
    bt_rl = run_backtest(df, signals_from_actions(df, rl_eval, cfg),
                         horizon=args.horizon, cost_r=args.cost_r)
    conv = [f"{a.epochs_run}ep{'*' if a.converged else ''}" for a in rl_agents]
    print(f"      RL folds converged: {conv} | {bt_rl.summary()}")

    print("[5/6] distill + generate MQL5 EA")
    dm = distill(df, lab, full_proba=res.proba)
    ea_path = out_dir / f"Pipeline_{tag}.mq5"
    code = render_ea(
        dm, symbol=args.symbol, provider=args.provider, timeframe=args.timeframe,
        call_th=call_th, put_th=put_th, sl_atr=args.sl_atr, rr=args.rr,
        horizon=args.horizon, full_auc=res.auc, bt_summary=bt.summary(),
        risk_pct=args.risk_pct,
    )
    ea_path.write_text(code)
    print(f"      distilled wf AUC {dm.wf_auc:.4f} | corr with full model {dm.corr_with_full:.3f}")
    print(f"      wrote {ea_path}")

    print("[6/6] report")
    top = res.feature_importance.head(20) if res.feature_importance is not None else pd.Series(dtype=float)
    fam_imp = (res.feature_importance.groupby(group_of).sum().sort_values(ascending=False)
               if res.feature_importance is not None else pd.Series(dtype=float))
    fam_dict = {k: int(v) for k, v in fam_imp.items()} if len(fam_imp) else "n/a"
    report = f"""# Pipeline run: {args.provider}:{args.symbol} {args.timeframe}

Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Data: {len(df)} bars
{df.index.min():%F} .. {df.index.max():%F}. Label: triple-barrier
(horizon {args.horizon} bars, SL {args.sl_atr} ATR, RR {args.rr}), costs {args.cost_r} R/round-trip.

## Behaviour across window permutations
{win_df.to_markdown(index=False)}

## Full-window model
- Purged walk-forward AUC: **{res.auc:.4f}** (folds {[round(a, 3) for a in res.auc_by_fold]})
- Features kept by relevance: {len(res.kept_features)} / {len(X.columns)}
- Family importance (gain): {fam_dict}

## Held-out backtest (second half of OOS, net of costs)
{json.dumps(bt.summary(), indent=2)}

Thresholds (chosen on first OOS half only): CALL >= {call_th:.2f}, PUT <= {put_th:.2f}.

## Recursive error-boosted refit (validation-early-stopped)
Validation AUC per round: {[round(h, 4) for h in rec.history]} — kept round
{rec.best_round}. Recursion stops when validation stops improving; training
error is never the stop criterion (that would just memorise the past).

## RL agent (contextual Q-learning, reward = net R, punishment = losses)
State = (model-probability bin, trend regime, volatility bin); actions
CALL/PUT/FLAT; trained by replaying past folds until the Q-table is stable
(fold epochs: {conv}, * = converged), then evaluated on the same held-out
half as the threshold policy:
{json.dumps(bt_rl.summary(), indent=2)}

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **{dm.wf_auc:.4f}**
- Spearman corr with full model OOS probabilities: {dm.corr_with_full:.3f}
- EA: `{ea_path.name}` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
{top.round(1).to_markdown()}

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
"""
    (out_dir / "report.md").write_text(report)
    res.proba.rename("proba").to_frame().to_parquet(out_dir / "oos_predictions.parquet")
    if not bt.trades.empty:
        bt.trades.to_parquet(out_dir / "trades.parquet")
    pd.Series(res.kept_features).to_csv(out_dir / "kept_features.csv", index=False)
    print(f"      wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
