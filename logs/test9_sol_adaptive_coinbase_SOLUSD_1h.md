# Test run: test9_sol_adaptive (coinbase_SOLUSD_1h)

**Prompt:** Test 9: try SOL with the live-correction approach (rolling in-window retraining + circuit breaker), same setup as test 6

**Run at:** 2026-07-12T21:04:12+00:00  |  **Window:** 2026-05-12 20:00:00+00:00 -> 2026-07-12 20:00:00+00:00 (1465 bars)

| Parameter | Value |
|---|---|
| Start balance | 6,000 |
| Leverage | 1.0:1 |
| Risk per trade | 1.0% |
| Label geometry | SL 1.5 ATR, RR 1.5, horizon 24 bars |
| Costs | 0.05 R/round-trip |
| Walk-forward AUC | 0.5248 |

## Results

| Policy | Final balance | Return | Trades | Win rate | Max DD |
|---|---|---|---|---|---|
| Threshold (CALL>=0.70/PUT<=0.30) | 4,862.08 | -18.97% | 77 | 29.9% | 18.97% |
| RL agent (converged=True) | 7,205.54 | 20.09% | 113 | 49.6% | 11.11% |
| Buy & hold | 4,911.74 | -18.14% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
