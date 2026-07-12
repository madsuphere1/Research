"""Live pattern-match engine: "which pattern does the current bar fit, and
how accurate has that pattern been HERE, in THIS regime?"

Every directional event feature in the pipeline (classical patterns, SMC
events, candle patterns, structure breaks, divergences, squeeze releases)
is treated as a *pattern signal*: its sign is the direction it implies.
For each signal we maintain an expanding, strictly causal ledger of past
occurrences and whether price went the implied way within `horizon` bars —
kept SEPARATELY per market regime (bear / chop / bull from the causal
trend+HMM state). At any bar the engine reports:

    * which signals are active right now
    * each one's historical hit rate in the CURRENT regime (+ sample size)
    * a combined direction verdict weighted by (edge x log evidence)

So after a profitable bullish trade, if a bearish pattern confirms on the
next bar, the verdict flips on that bar — the regime and pattern stats are
re-evaluated per bar, not per training window.

CLI:  python -m pipeline.match --provider coinbase --symbol BTC-USD \
          --timeframe 1h --years 2 [--test-name live1 --prompt "..."]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_ohlcv
from .features import build_features
from .features.trend import atr

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

# every signed event feature the engine treats as a "pattern signal"
PATTERN_SIGNALS = [
    # classical chart patterns
    "pat_double_top", "pat_double_bottom", "pat_triple_top", "pat_triple_bottom",
    "pat_hs", "pat_inv_hs", "pat_asc_triangle", "pat_desc_triangle",
    "pat_rising_wedge", "pat_falling_wedge", "pat_channel", "pat_flag",
    # market structure events
    "ms_int_bos_dir", "ms_int_choch_dir", "ms_int_mss",
    "ms_ext_bos_dir", "ms_ext_choch_dir", "ms_ext_mss",
    # SMC events
    "smc_fvg_dir", "smc_ob_dir", "smc_breaker", "smc_sweep_dir",
    "smc_int_sweep", "smc_ote_zone",
    # candle patterns
    "cdl_hammer", "cdl_shooting_star", "cdl_engulfing", "cdl_harami",
    "cdl_star", "cdl_three_soldiers_crows", "cdl_marubozu",
    # momentum / volatility events
    "mom_rsi_div", "mom_rsi_hidden_div", "mom_macd_div", "mom_macd_hidden_div",
    "mom_exhaustion", "vlt_squeeze_release", "sr_flip",
]

N_REGIMES = 3  # 0 bear, 1 chop, 2 bull


def regime_series(X: pd.DataFrame) -> pd.Series:
    """Per-bar regime from causal features: trend structure + HMM drift."""
    score = np.sign(X.get("trend_regime", 0)).fillna(0)
    if "ts_hmm_bull" in X and X["ts_hmm_bull"].notna().any():
        hmm = (X["ts_hmm_bull"].fillna(1 / 3) - X["ts_hmm_bear"].fillna(1 / 3))
        score = score + np.sign(hmm.where(hmm.abs() > 0.2, 0))
    return pd.Series(
        np.where(score > 0, 2, np.where(score < 0, 0, 1)), index=X.index
    )


def conditional_stats(
    X: pd.DataFrame,
    close: pd.Series,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    """Causal expanding hit-rate ledger per (signal, regime).

    Returns (stats long-frame with one row per signal x regime holding the
    FINAL ledger state, and the regime series). For live scoring only the
    final state matters; for research the per-bar ledgers stay causal
    because counts at bar t only include events resolved before t.
    """
    reg = regime_series(X)
    fwd = np.sign(close.shift(-horizon) - close).reindex(X.index)
    rows = []
    for col in PATTERN_SIGNALS:
        if col not in X:
            continue
        v = X[col].fillna(0.0)
        # count each activation once (entry into non-zero state)
        new_event = (v != 0) & ((v.shift(1) != v) | (v.shift(1) == 0))
        ev = X.index[new_event & fwd.notna()]
        for r in range(N_REGIMES):
            mask = reg.loc[ev] == r
            e = ev[mask]
            if len(e) == 0:
                rows.append({"signal": col, "regime": r, "n": 0, "hit_rate": np.nan})
                continue
            hits = (np.sign(v.loc[e]) == fwd.loc[e]).mean()
            rows.append({"signal": col, "regime": r, "n": int(len(e)),
                         "hit_rate": round(float(hits), 4)})
    return pd.DataFrame(rows), reg


def live_snapshot(
    df: pd.DataFrame,
    X: pd.DataFrame,
    stats: pd.DataFrame,
    reg: pd.Series,
    horizon: int = 24,
    sl_atr: float = 1.5,
    rr: float = 1.5,
    min_n: int = 10,
) -> dict:
    """Rank the patterns active on the LAST bar by regime-conditional
    accuracy and combine them into a CALL/PUT/FLAT verdict."""
    t = X.index[-1]
    cur_reg = int(reg.iloc[-1])
    reg_name = {0: "bear", 1: "chop/neutral", 2: "bull"}[cur_reg]
    active = []
    score = 0.0
    for col in PATTERN_SIGNALS:
        if col not in X:
            continue
        v = float(X[col].iloc[-1]) if pd.notna(X[col].iloc[-1]) else 0.0
        if v == 0:
            continue
        row = stats[(stats.signal == col) & (stats.regime == cur_reg)]
        n = int(row.n.iloc[0]) if len(row) else 0
        hr = float(row.hit_rate.iloc[0]) if len(row) and n > 0 else np.nan
        edge = (hr - 0.5) if (n >= min_n and np.isfinite(hr)) else 0.0
        w = np.log1p(n)
        score += np.sign(v) * edge * w
        active.append({
            "signal": col, "direction": "bullish" if v > 0 else "bearish",
            "hit_rate_in_regime": None if not np.isfinite(hr) else hr,
            "n_in_regime": n,
            "reliable": bool(n >= min_n),
        })
    active.sort(key=lambda a: -(a["hit_rate_in_regime"] or 0) * a["n_in_regime"])

    a = float(atr(df).iloc[-1])
    px = float(df.close.iloc[-1])
    verdict = "CALL" if score > 0.05 else ("PUT" if score < -0.05 else "FLAT")
    plan = None
    if verdict != "FLAT":
        s = 1 if verdict == "CALL" else -1
        plan = {"entry~": px, "sl": round(px - s * sl_atr * a, 2),
                "tp": round(px + s * rr * sl_atr * a, 2), "rr": rr}
    return {
        "bar_time": str(t),
        "regime": reg_name,
        "n_active_patterns": len(active),
        "active_patterns": active,
        "direction_score": round(float(score), 4),
        "verdict": verdict,
        "trade_plan": plan,
        "note": "hit rates are historical, regime-conditional, causal; "
                f"direction measured over next {horizon} bars",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="coinbase")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--test-name", default=None)
    ap.add_argument("--prompt", default="")
    args = ap.parse_args(argv)

    print(f"[1/3] loading {args.provider}:{args.symbol} {args.timeframe}")
    df = load_ohlcv(args.provider, args.symbol, args.timeframe, years=args.years)
    print(f"[2/3] features + causal per-regime pattern ledgers ({len(df)} bars)")
    X = build_features(df)
    stats, reg = conditional_stats(X, df.close.reindex(X.index), horizon=args.horizon)
    print("[3/3] live snapshot")
    snap = live_snapshot(df, X, stats, reg, horizon=args.horizon)
    print(json.dumps(snap, indent=2))

    # top table: most reliable patterns for this instrument per regime
    best = (stats[stats.n >= 10]
            .assign(edge=lambda d: (d.hit_rate - 0.5).abs())
            .sort_values("edge", ascending=False).head(15))
    print("\nMost informative patterns for this instrument (|hit-50%|, n>=10):")
    print(best.to_string(index=False))

    if args.test_name:
        LOGS.mkdir(exist_ok=True)
        rec = {"test_name": args.test_name, "prompt": args.prompt,
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "params": vars(args), "snapshot": snap,
               "top_patterns": best.to_dict("records")}
        with open(LOGS / "test_runs.jsonl", "a") as f:
            f.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        stats.to_csv(LOGS / f"{args.test_name}_pattern_stats.csv", index=False)
        print(f"\nlogged -> logs/test_runs.jsonl + {args.test_name}_pattern_stats.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
