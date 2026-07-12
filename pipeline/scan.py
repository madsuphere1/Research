"""Multi-timeframe predictability scan: not every possibility — the
standard ladder (15m / 1h / 6h / 1d), each measured with the same purged
walk-forward, so the instrument is traded on the timeframe where it is
actually most predictable, not where we assumed.

    python -m pipeline.scan --provider coinbase --symbol BTC-USD \
        [--test-name scan1 --prompt "..."]

Horizons scale with the timeframe (~1 trading day of bars).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import load_ohlcv
from .features import build_features
from .labeling import triple_barrier
from .model import walk_forward
from .signal import SignalConfig, choose_thresholds, make_signals

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

# timeframe -> (years of data, label horizon in bars)
LADDER = {"15m": (0.75, 32), "1h": (2.0, 24), "6h": (3.0, 12), "1d": (4.0, 8)}


def scan_timeframe(provider, symbol, tf, years, horizon, sl_atr, rr, cost_r):
    df = load_ohlcv(provider, symbol, tf, years=years)
    if len(df) < 3000:
        return {"timeframe": tf, "bars": len(df), "error": "not enough bars"}
    X = build_features(df)
    lab = triple_barrier(df, horizon=horizon, sl_atr=sl_atr, rr=rr).loc[X.index]
    res = walk_forward(X, lab, n_folds=4, purge=horizon)
    oos = res.proba.dropna()
    half = oos.index[len(oos) // 2]
    call_th, put_th = choose_thresholds(res.proba[res.proba.index < half],
                                        lab["r_long"], rr=rr, cost_r=cost_r)
    cfg = SignalConfig(rr=rr, sl_atr=sl_atr, call_th=call_th, put_th=put_th, cost_r=cost_r)
    bt = run_backtest(df, make_signals(df, res.proba[res.proba.index >= half], cfg),
                      horizon=horizon, cost_r=cost_r)
    return {
        "timeframe": tf, "bars": len(df), "horizon_bars": horizon,
        "wf_auc": round(res.auc, 4),
        "folds_above_half": f"{sum(a > 0.5 for a in res.auc_by_fold)}/{len(res.auc_by_fold)}",
        "held_out_net_r": bt.summary()["net_r_per_trade"],
        "n_trades": bt.summary()["n_trades"],
        "p_value": bt.summary()["p_value"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="coinbase")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--cost-r", type=float, default=0.05)
    ap.add_argument("--timeframes", default="15m,1h,6h,1d")
    ap.add_argument("--test-name", default=None)
    ap.add_argument("--prompt", default="")
    args = ap.parse_args(argv)

    rows = []
    for tf in args.timeframes.split(","):
        years, horizon = LADDER[tf]
        print(f"--- scanning {tf} ({years}y, horizon {horizon} bars)")
        try:
            row = scan_timeframe(args.provider, args.symbol, tf, years, horizon,
                                 args.sl_atr, args.rr, args.cost_r)
        except Exception as e:  # keep scanning other timeframes
            row = {"timeframe": tf, "error": str(e)[:120]}
        rows.append(row)
        print(f"    {row}")

    table = pd.DataFrame(rows)
    ok = table[table.get("wf_auc").notna()] if "wf_auc" in table else pd.DataFrame()
    best = (ok.sort_values(["held_out_net_r", "wf_auc"], ascending=False).iloc[0].to_dict()
            if len(ok) else None)
    print("\n=== timeframe ranking ===")
    print(table.to_string(index=False))
    if best:
        print(f"\nMost predictable timeframe for {args.symbol}: {best['timeframe']} "
              f"(AUC {best['wf_auc']}, held-out net R {best['held_out_net_r']})")

    if args.test_name:
        LOGS.mkdir(exist_ok=True)
        rec = {"test_name": args.test_name, "prompt": args.prompt,
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "symbol": args.symbol, "provider": args.provider,
               "ranking": rows, "best": best}
        with open(LOGS / "test_runs.jsonl", "a") as f:
            f.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        table.to_csv(LOGS / f"{args.test_name}_timeframe_scan.csv", index=False)
        print(f"logged -> logs/{args.test_name}_timeframe_scan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
