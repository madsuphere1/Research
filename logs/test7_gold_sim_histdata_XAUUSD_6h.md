# Test run: test7_gold_sim (histdata_XAUUSD_6h)

**Prompt:** Test 7: Gold, 6k, 1:10 leverage, last 2 months, on the most predictive timeframe (6h)

**Run at:** 2026-07-12T20:44:51+00:00  |  **Window:** 2026-04-26 18:00:00+00:00 -> 2026-06-26 18:00:00+00:00 (188 bars)

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
| Threshold (CALL>=0.52/PUT<=0.48) | 6,072.93 | 1.22% | 23 | 47.8% | 5.57% |
| RL agent (converged=True) | 5,476.24 | -8.73% | 25 | 28.0% | 10.03% |
| Buy & hold | 5,238.06 | -12.7% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
