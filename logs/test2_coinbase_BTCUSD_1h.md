# Test run: test2 (coinbase_BTCUSD_1h)

**Prompt:** Test 2: same as test 1 but the period is 2019-2020

**Run at:** 2026-07-12T19:56:59+00:00  |  **Window:** 2019-01-01 00:00:00+00:00 -> 2020-12-31 00:00:00+00:00 (17515 bars)

| Parameter | Value |
|---|---|
| Start balance | 200,000 |
| Leverage | 1.0:1 |
| Risk per trade | 1.0% |
| Label geometry | SL 1.5 ATR, RR 1.5, horizon 24 bars |
| Costs | 0.05 R/round-trip |
| Walk-forward AUC | 0.5491 |

## Results

| Policy | Final balance | Return | Trades | Win rate | Max DD |
|---|---|---|---|---|---|
| Threshold (CALL>=0.70/PUT<=0.30) | 167,363.21 | -16.32% | 994 | 42.7% | 35.77% |
| RL agent (converged=True) | 51,118.70 | -74.44% | 1562 | 39.6% | 75.48% |
| Buy & hold | 1,577,739.32 | 688.87% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
