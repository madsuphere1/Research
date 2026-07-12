# Super-Research: Behaviour-Driven Market Prediction Pipeline

Goal: a **symbol-agnostic research pipeline** that, given any instrument
(stock / forex / crypto / metal), downloads its history, studies its
behaviour across permuted lookback windows, learns which market-structure /
pattern / volume / volatility metrics actually carry signal **for that
instrument**, trains a walk-forward ML model, emits CALL / PUT decisions
with take-profit, stop-loss and risk:reward, and generates a ready-to-compile
**MQL5 Expert Advisor** for MetaTrader 5.

Related prior work: `../Claude-researcg` (XAUUSD 15m edge-existence study,
cycles 1–2). Key lesson carried over: **always cost-adjust, always
walk-forward, never trust in-sample results.**

## Phases

| Phase | Deliverable | Status tracker |
|---|---|---|
| 0 | Workspace, reference repos cloned (`external/`), reading rules (`docs/MARKET_RULES.md`) | PROGRESS.md |
| 1 | Data layer: multi-provider loader (Coinbase, Bitstamp, Kraken, HistData forex, local CSV/MT5 export) + window-permutation engine (6m / 1y / 2y lookbacks, shifted offsets) | tests/test_data.py |
| 2 | Feature engine: market structure (HH/HL/LH/LL, BOS, CHoCH, MSS), trend regimes, SMC (FVG, order blocks, liquidity sweeps, EQH/EQL, premium/discount), candle geometry + patterns, volume, volatility, momentum divergence, S/R zones, session/time | tests/test_features.py |
| 3 | Triple-barrier labeling, purged walk-forward LightGBM, per-instrument feature-relevance selection (metrics that don't help an instrument are dropped automatically) | tests/test_model.py |
| 4 | Signal engine: CALL/PUT + ATR-based TP/SL + risk:reward; cost-aware backtest | tests/test_signal.py |
| 5 | MQL5 generator: compilable `.mq5` EA parameterised by learned model output | tests/test_mql5.py |
| 6 | End-to-end runs on ≥2 different exchanges + window-permutation research report | reports/ |

## Design rules

1. **Be mindful of what and where**: third-party repos live in `external/`
   (gitignored — re-clone with `scripts/clone_references.sh`); our code in
   `pipeline/`; run artifacts in `outputs/` (gitignored) with committed
   summary reports in `reports/`.
2. Every phase has tests; a phase is done only when its tests pass.
3. Not every metric suits every instrument — Phase 3 measures per-instrument
   feature relevance and the signal/MQL5 stages only use what survived.
4. All evaluation is walk-forward and cost-aware. No look-ahead: features at
   bar *t* use data ≤ *t*; labels use data > *t*.
