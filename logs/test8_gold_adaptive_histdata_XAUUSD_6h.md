# Test run: test8_gold_adaptive (histdata_XAUUSD_6h)

**Prompt:** Test 8: gold rerun of test 7 with live correction ON (rolling in-window retraining + circuit breaker) — why didn't you stop the trade / recursively correct the prediction

**Run at:** 2026-07-12T21:01:26+00:00  |  **Window:** 2026-04-26 18:00:00+00:00 -> 2026-06-26 18:00:00+00:00 (188 bars)

| Parameter | Value |
|---|---|
| Start balance | 6,000 |
| Leverage | 10.0:1 |
| Risk per trade | 1.0% |
| Label geometry | SL 1.5 ATR, RR 1.5, horizon 12 bars |
| Costs | 0.05 R/round-trip |
| Walk-forward AUC | 0.4991 |

## Results

| Policy | Final balance | Return | Trades | Win rate | Max DD |
|---|---|---|---|---|---|
| Threshold (CALL>=0.52/PUT<=0.48) | 6,097.08 | 1.62% | 24 | 50.0% | 3.18% |
| RL agent (converged=True) | 6,047.28 | 0.79% | 17 | 47.1% | 4.88% |
| Buy & hold | 5,238.06 | -12.7% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
