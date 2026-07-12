# Research summary — two-exchange pipeline validation (2026-07-12)

The full pipeline (data → window permutations → 143-column feature matrix →
shadow relevance → purged walk-forward LightGBM → threshold selection →
held-out cost-aware backtest → MQL5 EA generation) was run end-to-end on
two different exchanges:

| | Coinbase BTC-USD 1h | Bitstamp ETH-USD 1h |
|---|---|---|
| Bars (2y) | 17,510 | 17,521 |
| Walk-forward AUC (full window) | 0.500 | 0.525 |
| AUC by window (6m/12m permutations) | 0.510 – 0.582 | 0.488 – 0.592 |
| Features kept by relevance | 63 / 143 | 73 / 143 |
| Held-out backtest net R/trade | −0.065 | −0.025 |
| Backtest p-value | 0.90 | 0.69 |
| Distilled (MQL5) walk-forward AUC | 0.517 | 0.534 |

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
4. **No tradeable edge after costs on these two crypto pairs at 1h.**
   Predictive signal above chance exists in several windows, but the
   held-out, cost-adjusted expectancy is negative and statistically
   indistinguishable from noise (p ≫ 0.05). This mirrors the sister study
   `../Claude-researcg` on XAUUSD 15m (real but small signal; dies at
   retail costs). The generated EAs carry these numbers in their headers
   deliberately: **they are research artifacts, not money printers.**

## Where to go next

- Lower-cost venues / maker-only execution (the sister repo's cycle-2
  found passive fills to be the single biggest lever).
- Forex/metals via the `histdata` provider (session features should
  matter more there), e.g.
  `python -m pipeline.run --provider histdata --symbol XAUUSD --timeframe 15m --years 2`.
- Higher timeframes (4h/1d) where the cost fraction per trade shrinks.
- Plug an options-chain provider into `features/options_stub.py` to light
  up the §12 feature family.
