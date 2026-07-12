# Testing protocol & test ledger

Every user-requested test is executed with full computation (no cached
feature matrices, no skipped stages — see AGENT GUIDE in CLAUDE.md), on
**one exchange at a time**, and persisted:

* `logs/test_runs.jsonl` — one JSON record per test: the user's prompt
  verbatim, all parameters, window, model quality, results.
* `logs/<test-name>_<tag>.md` — human-readable result sheet.
* `logs/<test-name>_*_trades_*.csv` — every simulated trade.

## How to run a test

```bash
# account simulation (balance / leverage / period / instrument)
python -m pipeline.simulate --provider coinbase --symbol SOL-USD \
    --balance 6000 --leverage 1 --days 61 \
    --test-name test6_sol --prompt "<user's words>"
# explicit historical window:
#   --start 2019-01-01 --end 2020-12-31

# which candle size is predictable for an instrument
python -m pipeline.scan --provider histdata --symbol XAUUSD \
    --test-name test7_gold_scan --prompt "<user's words>"

# what does the live bar match right now (regime-conditional accuracy)
python -m pipeline.match --provider coinbase --symbol BTC-USD \
    --timeframe 1h --test-name test3_live_match --prompt "<user's words>"
```

Honesty invariants (enforced in code, never to be weakened):
predictions inside a simulated window are walk-forward out-of-sample;
every policy's calibration is fitted only on pre-window data **from its
own probability source** (test 9 lesson: cutoffs calibrated on one model's
distribution must never filter another model's output); costs are always
included; ties resolve SL-first; entries fill at next bar open.

**Where live correction is used (evidence-based, since test 10):** the
simulator runs three policies side by side — `threshold_static` (static
model + static-fitted cutoffs), `threshold_adaptive_recalibrated`
(rolling-retrained model + cutoffs fitted on a pre-window segment of
*adaptive* predictions), and `rl_adaptive` (RL states over the
live-corrected probabilities — the pairing that improved every instrument
tried). The circuit breaker guards all three.

## Ledger (as of 2026-07-12)

| # | Ask | Setup | Outcome (best policy vs buy & hold) |
|---|---|---|---|
| 1 | "200k last year, BTCUSD, 1:1 — value now?" | $200k, 365d, BTC 1h | **$167,479 (RL)** vs $109,001 B&H — cushioned a −45% market |
| 2 | "Same but 2019–2020" | $200k, 2019-01-01→2020-12-31 | $167,363 (thr) vs **$1,577,739 B&H** — badly lagged the bull; RL (bear-trained) lost 74% |
| 3 | "Which of the ~70 patterns does live data fit, with what accuracy?" | BTC 1h live match | 12 active patterns ranked; regime flips pattern meaning (ext-CHoCH 71% in chop, 32% in bear) |
| 4 | "Which timeframe is most predictable?" | BTC scan 15m/1h/6h/1d | **6h: AUC 0.564, +0.177 R/trade, p=0.024** (project's best; ~0.10 after multiple-test correction) |
| 5 | "Any other exchange — Solana?" | SOL scan + match + full run | Best TF 1h; breakouts fail on SOL (fade); +0.054 R held-out, p=0.12; EA generated |
| 6 | "6k two months ago — now?" | $6k, 61d, SOL 1h, 1:1 | **$7,300 (+21.7%)** vs $4,912 B&H (−18%) |
| 7 | "Gold 2 months, which candle size, 6k at 1:10" | XAUUSD scan + $6k 6h sim | 6h most predictive (AUC 0.550, 4/4) but no TF beats costs; sim $6,073 (+1.2%) vs $5,238 B&H |
| 8 | "Why didn't you stop the trade / recursively correct the prediction?" | Test 7 rerun, live correction ON | Threshold $6,097 (+1.6%, max DD 5.6%→3.2%); **RL −8.7% → +0.8%** — breaker tripped once, skipped 70 signals |
| 9 | "Try SOL with this approach" | Test 6 rerun, live correction ON | Split verdict: **RL +7.4% → +20.1%** ($7,206, best run yet) but threshold +21.7% → **−19.0%** — thresholds calibrated on the static model's probabilities don't fit the retrained model's distribution (known issue below); breaker capped the damage (5 trips, 376 signals skipped) |
| 10 | "Now you know where to use live correction — update code accordingly" | SOL rerun, three paired policies | **threshold_static restored: +19.4%**; threshold_adaptive recalibrated −19.0% → −4.4% (better but still worst on SOL); rl_adaptive +3.5% — the drop from test 9's +20.1% came only from shifting the retrain phase, so RL-adaptive results are **unstable across retrain timing**; all three beat B&H (−18.1%) |

## What the ledger shows so far

1. The system reliably **loses less than the market in down moves**
   (tests 1, 6, 7) and reliably **lags strong bull trends** (test 2).
2. Policies are **regime-dependent**: the RL agent is strong when the
   simulated regime resembles its training regime and dangerous otherwise.
3. **Timeframe is per-instrument**: BTC 6h, SOL 1h, gold 6h-ish.
4. Two-month windows are weather, not climate — judge the system on the
   whole ledger, not one row.
5. **Live correction is instrument-dependent** (tests 8–10): it rescued
   gold's RL policy, but on SOL the static threshold policy remains best
   and the rolling-retrained model underperforms for threshold filtering.
   RL-adaptive returns vary a lot with retrain timing (+20.1% vs +3.5% for
   the same window) — treat single adaptive runs with extra suspicion.
   This is why the simulator reports all three policies side by side:
   the per-instrument evidence accumulates in this ledger instead of being
   decided by one lucky configuration.

## Changes made because of testing (chronological)

* Test 1 exposed a JSON-serialisation crash on numpy bools → fixed.
* Test 2 required explicit historical windows → `--start/--end` added.
* Test 2's RL collapse motivated the regime-conditional pattern ledger
  (`pipeline/match.py`) — posture now re-evaluated every bar.
* "Infinite timeframes" concern → bounded ladder scan (`pipeline/scan.py`).
* Slow repeat runs → feature-matrix cache added, then **reverted at the
  user's request** ("don't bypass any step"); every run computes fresh.
* Parallel runs slowed each other → tests now run one exchange at a time.
* Test 7's bad gold trades ("why didn't you stop / correct?") → simulator
  gained (a) a **circuit breaker** (pause after a −4R 10-trade streak,
  cooldown, resume) and (b) **in-window rolling retraining**
  (`adaptive_proba`: refit every N bars on the latest resolved bars; the
  rolling window is essential — an expanding window never unlearns a
  flipped regime, proven by `tests/test_adaptive.py`). Both defaults-on;
  every test trains a fresh model, stated in each log record.
* Test 9 exposed a calibration mismatch: CALL/PUT thresholds are fitted on
  the STATIC pre-window model's probability distribution but applied to
  the retrained models' probabilities, whose distribution shifts.
  Coarse-binned RL states are robust to this (RL improved on both gold and
  SOL); fixed thresholds are not (SOL threshold policy flipped from +21.7%
  to −19.0%). Open fix: generate pre-window predictions with the same
  rolling-retrain procedure and fit thresholds on those.
