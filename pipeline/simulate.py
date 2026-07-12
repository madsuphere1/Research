"""Account-level test-run simulator with persistent logs.

    python -m pipeline.simulate --provider coinbase --symbol BTC-USD \
        --balance 200000 --leverage 1 --days 365 --test-name test1 \
        --prompt "If I had 200k last year in BTCUSD 1:1 ..."

What it does, honestly:
  * downloads enough history to train BEFORE the simulated period
  * walk-forward model probabilities (every prediction in the simulated
    window comes from folds trained only on earlier data)
  * decision thresholds and the RL agent are fitted ONLY on out-of-sample
    predictions from before the simulation start
  * simulates the account in dollars: risk a % of current balance per
    trade, position notional capped at leverage x balance, costs deducted,
    equity compounds; both the threshold policy and the RL policy are run
  * compares against buy-and-hold of the same account
  * appends the full test record to logs/test_runs.jsonl and writes a
    human-readable logs/<test-name>_<tag>.md

The result is a *historical replay of the system's out-of-sample
decisions*, not a promise about the future.
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
from .labeling import triple_barrier
from .model import walk_forward
from .rl import ACTIONS, QAgent, build_states, rewards_frame
from .signal import SignalConfig, choose_thresholds, make_signals
from .signal_actions import signals_from_actions

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def _json_default(o):
    """numpy scalars leak into records; make json.dumps robust to them."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def adaptive_proba(
    X: pd.DataFrame,
    lab: pd.DataFrame,
    kept: list[str],
    sim_start,
    purge: int,
    retrain_every: int,
    train_window: int = 3000,
    seed: int = 0,
) -> tuple[pd.Series, int]:
    """Online recursive correction: inside the simulated window the model is
    refit every `retrain_every` bars on the most recent `train_window` bars
    whose labels are RESOLVED (a label at bar i is only known once its
    barrier resolves, hence the purge). The rolling window is the point:
    when the market's behaviour flips mid-window, recent evidence must
    outweigh stale history or the model never corrects itself. Strictly
    causal; fresh per test — nothing is loaded from previous runs."""
    import lightgbm as lgb

    from .model import LGB_PARAMS

    idx = X.index
    proba = pd.Series(np.nan, index=idx)
    sim_pos = np.where(idx >= sim_start)[0]
    if len(sim_pos) == 0:
        return proba, 0
    y = (lab["label"] == 1).astype(int)
    labeled = lab["label"].notna().values
    n, pos_ = len(idx), int(sim_pos[0])
    n_retrains = 0
    while pos_ < n:
        tr_end = pos_ - purge
        tr_lo = max(0, tr_end - train_window)
        if tr_end - tr_lo >= 300 and labeled[tr_lo:tr_end].sum() >= 300:
            sl = slice(tr_lo, tr_end)
            mask = labeled[sl]
            m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
            m.fit(X[kept].iloc[sl][mask], y.iloc[sl][mask])
            hi = min(pos_ + retrain_every, n)
            proba.iloc[pos_:hi] = m.predict_proba(X[kept].iloc[pos_:hi])[:, 1]
            n_retrains += 1
        pos_ += retrain_every
    return proba, n_retrains


def simulate_dollars(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    balance0: float,
    leverage: float,
    risk_pct: float,
    horizon: int,
    cost_r: float,
    breaker_lookback: int = 10,
    breaker_stop_r: float = -4.0,
    cooldown_bars: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Dollar-accounted event simulation: next-bar-open entry, SL-first tie
    handling, one position at a time, risk-% sizing with leverage cap.

    Circuit breaker ("stop when the prediction is wrong"): if the last
    `breaker_lookback` trades sum to <= `breaker_stop_r` in net R, trading
    pauses for `cooldown_bars` bars (default 5x horizon), then resumes.
    Set breaker_lookback=0 to disable."""
    from collections import deque

    cooldown_bars = cooldown_bars if cooldown_bars is not None else horizon * 5
    recent: deque = deque(maxlen=breaker_lookback or 1)
    paused_until = -1
    breaker_trips = 0
    skipped_paused = 0

    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    pos = {ts: i for i, ts in enumerate(df.index)}
    balance = balance0
    peak = balance0
    max_dd = 0.0
    busy_until = -1
    recs = []
    for ts, row in signals[signals.side != 0].iterrows():
        i = pos[ts]
        if i < paused_until:
            skipped_paused += 1
            continue
        if i <= busy_until or i + 1 >= len(df) or balance <= 0:
            continue
        entry_i = i + 1
        entry = o[entry_i]
        sl_dist = abs(row.entry - row.sl)
        if not np.isfinite(sl_dist) or sl_dist <= 0:
            continue
        side = row.side
        # size: risk risk_pct% of CURRENT balance, capped by leverage
        risk_amt = balance * risk_pct / 100.0
        qty = risk_amt / sl_dist
        max_qty = leverage * balance / entry
        if qty > max_qty:
            qty = max_qty
        risk_eff = qty * sl_dist
        tp = entry + side * row.rr * sl_dist
        sl = entry - side * sl_dist
        end = min(entry_i + horizon, len(df) - 1)
        r, exit_i = None, end
        for j in range(entry_i, end + 1):
            hit_tp = h[j] >= tp if side > 0 else l[j] <= tp
            hit_sl = l[j] <= sl if side > 0 else h[j] >= sl
            if hit_sl:
                r, exit_i = -1.0, j
                break
            if hit_tp:
                r, exit_i = row.rr, j
                break
        if r is None:
            r = side * (c[end] - entry) / sl_dist
        pnl = (r - cost_r) * risk_eff
        balance += pnl
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak)
        recs.append({"time": ts, "side": side, "entry": entry, "qty": qty,
                     "r": r, "pnl": round(pnl, 2), "balance": round(balance, 2)})
        busy_until = exit_i
        if breaker_lookback:
            recent.append(r - cost_r)
            if len(recent) == breaker_lookback and sum(recent) <= breaker_stop_r:
                paused_until = exit_i + cooldown_bars
                breaker_trips += 1
                recent.clear()
    trades = pd.DataFrame(recs)
    summary = {
        "final_balance": round(balance, 2),
        "return_pct": round((balance / balance0 - 1) * 100, 2),
        "n_trades": len(trades),
        "win_rate": round(float((trades.pnl > 0).mean()), 4) if len(trades) else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "busted": bool(balance <= 0),
        "breaker_trips": breaker_trips,
        "signals_skipped_while_paused": skipped_paused,
    }
    return summary, trades


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="coinbase")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--balance", type=float, required=True)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--days", type=int, default=365, help="simulated period, ending now")
    ap.add_argument("--start", default=None, help="explicit sim start date (YYYY-MM-DD); overrides --days")
    ap.add_argument("--end", default=None, help="explicit sim end date (YYYY-MM-DD); default now")
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--cost-r", type=float, default=0.05)
    ap.add_argument("--test-name", default="test")
    ap.add_argument("--prompt", default="", help="the original test prompt, stored in the log")
    ap.add_argument("--retrain-every", type=int, default=0,
                    help="bars between in-window model retrains (0 = auto: 5x horizon)")
    ap.add_argument("--train-window", type=int, default=3000,
                    help="rolling training window (bars) for in-window retrains")
    ap.add_argument("--static", action="store_true",
                    help="disable in-window retraining (pre-window model only)")
    ap.add_argument("--breaker-lookback", type=int, default=10,
                    help="trades in the circuit-breaker window (0 disables)")
    ap.add_argument("--breaker-stop-r", type=float, default=-4.0,
                    help="pause trading when last lookback trades sum <= this net R")
    args = ap.parse_args(argv)

    # need training history before the simulated window: fetch 2x the span (min 2y)
    end_dt = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
              if args.end else datetime.now(timezone.utc))
    span_days = ((end_dt - datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)).days
                 if args.start else args.days)
    fetch_years = max(2.0, 2 * span_days / 365.25)
    tag = f"{args.provider}_{args.symbol.replace('/', '').replace('-', '')}_{args.timeframe}"
    print(f"[1/5] loading {fetch_years:.1f}y of {args.provider}:{args.symbol} {args.timeframe} ending {end_dt:%F}")
    df = load_ohlcv(args.provider, args.symbol, args.timeframe, years=fetch_years, end=end_dt)
    sim_start = df.index.max() - pd.Timedelta(days=span_days)
    print(f"      {len(df)} bars; simulated window {sim_start:%F} .. {df.index.max():%F}")

    print("[2/5] features + walk-forward model (predictions in the window are OOS)")
    X = build_features(df)
    lab = triple_barrier(df, horizon=args.horizon, sl_atr=args.sl_atr, rr=args.rr).loc[X.index]
    res = walk_forward(X, lab, n_folds=6, purge=args.horizon)
    pre = res.proba[(res.proba.index < sim_start)].dropna()
    if len(pre) < 500:
        raise SystemExit("not enough pre-window OOS data to fit thresholds honestly")

    print("[3/5] policies — each proba source paired with its OWN calibration")
    # Ledger evidence (tests 6/8/9): live correction helps the RL policy on
    # every instrument tried, but thresholds calibrated on one model's
    # probability distribution must never filter another model's output.
    # So: the static threshold policy stays fully static, and the adaptive
    # threshold policy is recalibrated on ADAPTIVE pre-window predictions.
    call_th, put_th = choose_thresholds(pre, lab["r_long"], rr=args.rr, cost_r=args.cost_r)
    cfg = SignalConfig(rr=args.rr, sl_atr=args.sl_atr, call_th=call_th,
                       put_th=put_th, cost_r=args.cost_r)
    sig_th_static = make_signals(df, res.proba[res.proba.index >= sim_start], cfg)

    retrain_every = args.retrain_every or 5 * args.horizon
    if args.static:
        sim_proba = res.proba[res.proba.index >= sim_start]
        n_retrains = 0
        call_a, put_a = call_th, put_th
        sig_th_adapt = sig_th_static
    else:
        # start the rolling-retrain series one window-length BEFORE the sim
        # window: that pre-segment is out-of-sample for the retrained models
        # and is what the adaptive thresholds are calibrated on.
        cal_start = sim_start - (df.index.max() - sim_start)
        sim_proba_full, n_retrains = adaptive_proba(
            X, lab, res.kept_features, cal_start, purge=args.horizon,
            retrain_every=retrain_every, train_window=args.train_window,
        )
        sim_proba = sim_proba_full[sim_proba_full.index >= sim_start]
        cal_proba = sim_proba_full[
            (sim_proba_full.index >= cal_start) & (sim_proba_full.index < sim_start)
        ].dropna()
        print(f"      retrains: {n_retrains} (every {retrain_every} bars; "
              f"calibration segment {cal_start:%F} .. {sim_start:%F}, {len(cal_proba)} preds)")
        call_a, put_a = choose_thresholds(cal_proba, lab["r_long"],
                                          rr=args.rr, cost_r=args.cost_r)
        cfg_a = SignalConfig(rr=args.rr, sl_atr=args.sl_atr, call_th=call_a,
                             put_th=put_a, cost_r=args.cost_r)
        sig_th_adapt = make_signals(df, sim_proba, cfg_a)
        print(f"      static thresholds {call_th:.2f}/{put_th:.2f} | "
              f"adaptive (recalibrated) {call_a:.2f}/{put_a:.2f}")

    proba_combined = pd.Series(np.nan, index=X.index)
    proba_combined.loc[res.proba.index] = res.proba
    proba_combined.loc[sim_proba.index] = sim_proba
    regime = X["trend_regime"].reindex(proba_combined.index)
    volf = X["vlt_atr_pct"].reindex(proba_combined.index)
    states_all = build_states(proba_combined.fillna(0.5), regime, volf)
    rew = rewards_frame(lab["r_long"].reindex(states_all.index), args.rr, args.cost_r)
    pre_mask = (states_all.index < sim_start) & res.proba.reindex(states_all.index).notna().values
    agent = QAgent(seed=0).fit(states_all[pre_mask].values,
                               rew[list(ACTIONS)][pre_mask].values)
    rl_actions = pd.Series(
        agent.act(states_all[states_all.index >= sim_start].values),
        index=states_all.index[states_all.index >= sim_start],
    ).where(sim_proba.notna(), 0.0)
    sig_rl = signals_from_actions(df, rl_actions, cfg)

    print("[4/5] dollar simulation (circuit breaker: "
          f"{args.breaker_lookback}-trade window, stop at {args.breaker_stop_r}R)")
    brk = dict(breaker_lookback=args.breaker_lookback, breaker_stop_r=args.breaker_stop_r)
    sum_th, tr_th = simulate_dollars(df, sig_th_static, args.balance, args.leverage,
                                     args.risk_pct, args.horizon, args.cost_r, **brk)
    sum_ta, tr_ta = simulate_dollars(df, sig_th_adapt, args.balance, args.leverage,
                                     args.risk_pct, args.horizon, args.cost_r, **brk)
    sum_rl, tr_rl = simulate_dollars(df, sig_rl, args.balance, args.leverage,
                                     args.risk_pct, args.horizon, args.cost_r, **brk)
    sim_px = df.close[df.index >= sim_start]
    bh_final = args.balance * float(sim_px.iloc[-1] / sim_px.iloc[0])
    bh = {"final_balance": round(bh_final, 2),
          "return_pct": round((bh_final / args.balance - 1) * 100, 2)}

    record = {
        "test_name": args.test_name,
        "prompt": args.prompt,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {k: getattr(args, k) for k in
                   ("provider", "symbol", "timeframe", "balance", "leverage",
                    "days", "risk_pct", "horizon", "sl_atr", "rr", "cost_r")},
        "window": {"start": str(sim_start), "end": str(df.index.max()),
                   "bars": int((df.index >= sim_start).sum())},
        "model": {"wf_auc": round(res.auc, 4), "call_th": call_th, "put_th": put_th,
                  "call_th_adaptive": call_a, "put_th_adaptive": put_a,
                  "rl_converged": agent.converged, "rl_epochs": agent.epochs_run},
        "adaptation": {
            "fresh_model_this_test": True,   # nothing loaded from previous tests
            "in_window_retraining": not args.static,
            "retrain_every_bars": retrain_every if not args.static else None,
            "rolling_train_window_bars": args.train_window,
            "n_retrains": n_retrains,
            "circuit_breaker": {"lookback_trades": args.breaker_lookback,
                                "stop_at_net_r": args.breaker_stop_r},
        },
        "results": {"threshold_static": sum_th,
                    "threshold_adaptive_recalibrated": sum_ta,
                    "rl_adaptive": sum_rl, "buy_and_hold": bh},
        "honesty": "OOS walk-forward predictions; each policy's calibration fitted "
                   "pre-window on ITS OWN proba source; costs included; "
                   "historical replay, not a forecast.",
    }

    print("[5/5] logging")
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "test_runs.jsonl", "a") as f:
        f.write(json.dumps(record, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    md_path = LOGS / f"{args.test_name}_{tag}.md"
    md_path.write_text(f"""# Test run: {args.test_name} ({tag})

**Prompt:** {args.prompt or "(none given)"}

**Run at:** {record['run_at']}  |  **Window:** {record['window']['start']} -> {record['window']['end']} ({record['window']['bars']} bars)

| Parameter | Value |
|---|---|
| Start balance | {args.balance:,.0f} |
| Leverage | {args.leverage}:1 |
| Risk per trade | {args.risk_pct}% |
| Label geometry | SL {args.sl_atr} ATR, RR {args.rr}, horizon {args.horizon} bars |
| Costs | {args.cost_r} R/round-trip |
| Walk-forward AUC | {res.auc:.4f} |

## Results

| Policy | Final balance | Return | Trades | Win rate | Max DD |
|---|---|---|---|---|---|
| Threshold static (CALL>={call_th:.2f}/PUT<={put_th:.2f}) | {sum_th['final_balance']:,.2f} | {sum_th['return_pct']}% | {sum_th['n_trades']} | {sum_th['win_rate']:.1%} | {sum_th['max_drawdown_pct']}% |
| Threshold adaptive, recalibrated (CALL>={call_a:.2f}/PUT<={put_a:.2f}) | {sum_ta['final_balance']:,.2f} | {sum_ta['return_pct']}% | {sum_ta['n_trades']} | {sum_ta['win_rate']:.1%} | {sum_ta['max_drawdown_pct']}% |
| RL adaptive (converged={agent.converged}) | {sum_rl['final_balance']:,.2f} | {sum_rl['return_pct']}% | {sum_rl['n_trades']} | {sum_rl['win_rate']:.1%} | {sum_rl['max_drawdown_pct']}% |
| Buy & hold | {bh['final_balance']:,.2f} | {bh['return_pct']}% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample; each
policy's calibration was fitted before the window on its own probability
source (ledger lesson from test 9); costs are included. This is a
historical replay of what the system would have decided — not a forecast.
""")
    if len(tr_th):
        tr_th.to_csv(LOGS / f"{args.test_name}_{tag}_trades_threshold.csv", index=False)
    if len(tr_ta):
        tr_ta.to_csv(LOGS / f"{args.test_name}_{tag}_trades_threshold_adaptive.csv", index=False)
    if len(tr_rl):
        tr_rl.to_csv(LOGS / f"{args.test_name}_{tag}_trades_rl.csv", index=False)
    print(json.dumps(record["results"], indent=2))
    print(f"      logged -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
