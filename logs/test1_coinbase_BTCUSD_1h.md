# Test run: test1 (coinbase_BTCUSD_1h)

**Prompt:** If I had 200k in my account last year (365 days) in BTCUSD at 1:1 leverage, what would the account value be right now in your system?

**Run at:** 2026-07-12T15:27:42+00:00  |  **Window:** 2025-07-12 14:00:00+00:00 -> 2026-07-12 14:00:00+00:00 (8751 bars)

| Parameter | Value |
|---|---|
| Start balance | 200,000 |
| Leverage | 1.0:1 |
| Risk per trade | 1.0% |
| Label geometry | SL 1.5 ATR, RR 1.5, horizon 24 bars |
| Costs | 0.05 R/round-trip |
| Walk-forward AUC | 0.5077 |

## Results

| Policy | Final balance | Return | Trades | Win rate | Max DD |
|---|---|---|---|---|---|
| Threshold (CALL>=0.70/PUT<=0.30) | 143,167.64 | -28.42% | 562 | 38.8% | 30.9% |
| RL agent (converged=True) | 167,479.02 | -16.26% | 836 | 40.8% | 30.46% |
| Buy & hold | 109,000.74 | -45.5% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample
(walk-forward); thresholds and the RL policy were fitted only on data from
before the window; costs are included. This is a historical replay of what
the system would have decided — not a forecast of future returns.
