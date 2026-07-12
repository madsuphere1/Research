# Test run: test6_sol (coinbase_SOLUSD_1h)

**Prompt:** Test 6: If I had 6k two months ago (SOL-USD, 1:1), what would it be now?

**Run at:** 2026-07-12T20:38:14+00:00  |  **Window:** 2026-05-12 20:00:00+00:00 -> 2026-07-12 20:00:00+00:00 (1465 bars)

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
| Threshold (CALL>=0.70/PUT<=0.30) | 7,300.04 | 21.67% | 89 | 51.7% | 4.79% |
| RL agent (converged=True) | 6,446.55 | 7.44% | 134 | 44.8% | 14.03% |
| Buy & hold | 4,911.74 | -18.14% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
