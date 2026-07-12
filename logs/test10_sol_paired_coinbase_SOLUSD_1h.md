# Test run: test10_sol_paired (coinbase_SOLUSD_1h)

**Prompt:** Test 10: SOL rerun with evidence-based pairing — static thresholds on static proba, recalibrated thresholds on adaptive proba, RL on adaptive

**Run at:** 2026-07-12T21:09:45+00:00  |  **Window:** 2026-05-12 20:00:00+00:00 -> 2026-07-12 20:00:00+00:00 (1465 bars)

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
| Threshold static (CALL>=0.70/PUT<=0.30) | 7,161.20 | 19.35% | 79 | 51.9% | 5.42% |
| Threshold adaptive, recalibrated (CALL>=0.70/PUT<=0.30) | 5,738.95 | -4.35% | 86 | 39.5% | 8.15% |
| RL adaptive (converged=True) | 6,211.35 | 3.52% | 113 | 44.2% | 13.74% |
| Buy & hold | 4,911.74 | -18.14% | 1 | — | — |

*Honesty note:* every prediction inside the window is out-of-sample; each
policy's calibration was fitted before the window on its own probability
source (ledger lesson from test 9); costs are included. This is a
historical replay of what the system would have decided — not a forecast.
