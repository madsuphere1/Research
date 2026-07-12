# Research summary — two-exchange pipeline validation (2026-07-12)

The full pipeline (data → window permutations → 150-column feature matrix
incl. time-series models → shadow relevance → purged walk-forward LightGBM
→ recursive error-boosted refit → threshold + RL policies → held-out
cost-aware backtest → MQL5 EA generation) was run end-to-end on two
different exchanges (updated after phases 7–9):

| | Coinbase BTC-USD 1h | Bitstamp ETH-USD 1h |
|---|---|---|
| Bars (2y) | 17,510 | 17,521 |
| Walk-forward AUC (full window) | 0.509 | 0.547 |
| AUC by window (6m/12m/24m permutations) | 0.475 – 0.594 | 0.497 – 0.579 |
| Features kept by relevance | 88 / 150 | 82 / 150 |
| Held-out backtest net R/trade (threshold policy) | −0.047 | −0.015 |
| Held-out net R/trade (RL agent, same half) | **+0.025** | −0.004 |
| RL backtest p-value | 0.28 | 0.56 |
| Distilled (MQL5) walk-forward AUC | 0.517 | 0.534 |

## Phase 7–9 additions

* **Pattern recognition ≥95% per pattern** (`reports/PATTERN_BENCH.md`):
  detectors tuned from a 60.8% baseline to mean 98.3% accuracy on labeled
  textbook shapes, regression-guarded. Recognition ≠ prediction — the
  predictive worth of each pattern is still measured only out-of-sample.
* **Time-series models** (rolling AR forecasts, forward-filtered HMM
  regimes) joined the feature matrix and lifted ETH's walk-forward AUC
  from 0.525 → 0.547; BTC 0.500 → 0.509.
* **Recursive error-boosted refit**: retrains with mistakes up-weighted,
  keeps the best *validation* round (BTC kept round 4 of 7; ETH stopped
  immediately — extra recursion was hurting). Training error is never the
  stop criterion, because "loop until the prediction is true" on the past
  is just memorisation.
* **RL agent** (contextual Q; reward = realised net R, losses punish):
  Q-tables converged on all folds. On the identical held-out half it beat
  the threshold policy on both instruments and turned BTC positive
  (+0.025 R/trade, PF 1.045) — but p = 0.28 means this is **not yet
  statistically distinguishable from luck**. It is a promising lever, not
  a proven edge.

## What the research found

1. **Market structure is the strongest family on both instruments** —
   `ms_ext_bars_since_bos` / `ms_ext_bars_since_choch` top the gain
   rankings, matching the "market structure first" hypothesis in
   `docs/MARKET_RULES.md §1`. Volume behaviour and trend regime come next;
   session features matter even for 24/7 crypto (Asian-range width is a
   top-5 ETH feature).
2. **Candlestick patterns are near-worthless once structure is known**
   (lowest gain family on both instruments) — consistent with the view
   that geometry context beats named candle patterns.
3. **The metric mix is instrument-specific and time-unstable.** Kept
   feature counts and window AUCs differ across (lookback, offset)
   permutations — e.g. both instruments score their best AUC on the
   6-months-ending-6-months-ago window (BTC 0.582, ETH 0.592) and much
   lower on the most recent 6 months. This is why the pipeline re-runs
   relevance per instrument and per window instead of assuming one fixed
   feature set.
4. **No *proven* tradeable edge after costs on these two crypto pairs at
   1h.** Predictive signal above chance exists (ETH walk-forward AUC 0.547
   with 5/5 folds above 0.5), and the RL layer improved expectancy on both
   instruments, but no configuration yet clears statistical significance
   net of costs. This mirrors the sister study `../Claude-researcg` on
   XAUUSD 15m (real but small signal; dies at retail costs). The generated
   EAs carry these numbers in their headers deliberately: **they are
   research artifacts, not money printers.**

## Where to go next

- Lower-cost venues / maker-only execution (the sister repo's cycle-2
  found passive fills to be the single biggest lever).
- Forex/metals via the `histdata` provider (session features should
  matter more there), e.g.
  `python -m pipeline.run --provider histdata --symbol XAUUSD --timeframe 15m --years 2`.
- Higher timeframes (4h/1d) where the cost fraction per trade shrinks.
- Plug an options-chain provider into `features/options_stub.py` to light
  up the §12 feature family.
