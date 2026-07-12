# Research — behaviour-driven market prediction pipeline

A symbol-agnostic research pipeline: give it any instrument reachable
through its data providers and it downloads history, studies behaviour
across permuted lookback windows, learns **which** market concepts actually
carry signal for that instrument, trains a purged walk-forward ML model,
emits CALL/PUT decisions with ATR take-profit / stop-loss at a fixed
risk:reward, and generates a compilable **MQL5 Expert Advisor** whose
header carries the measured out-of-sample quality.

Sister repo: [`Claude-researcg`](../Claude-researcg) — the earlier XAUUSD
edge-existence study whose honesty rules (walk-forward only, cost-aware,
placebo-tested) this pipeline inherits.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/clone_references.sh      # optional: reference repos -> external/

python -m pipeline.run --provider coinbase --symbol BTC-USD --timeframe 1h --years 2
python -m pipeline.run --provider bitstamp --symbol ETH-USD --timeframe 1h --years 2
python -m pipeline.run --provider histdata --symbol XAUUSD  --timeframe 15m --years 2
python -m pipeline.run --provider csv --symbol path/to/mt5_export.csv --timeframe 1h

# account replay: what would $X have become over period P at leverage L?
python -m pipeline.simulate --provider coinbase --symbol SOL-USD \
    --balance 6000 --leverage 1 --days 61 --test-name test6 --prompt "..."
# which candle size is this instrument predictable on?
python -m pipeline.scan --provider histdata --symbol XAUUSD --test-name scan1
# what does the live bar match right now (regime-conditional accuracy)?
python -m pipeline.match --provider coinbase --symbol BTC-USD --timeframe 1h
```

Test protocol, honesty invariants and the ledger of user tests 1–7 are in
[`TESTING.md`](TESTING.md); agent operating rules in [`CLAUDE.md`](CLAUDE.md).

Outputs land in `outputs/<provider>_<symbol>_<tf>/`:
`report.md`, `window_research.csv`, `oos_predictions.parquet`,
`trades.parquet`, `kept_features.csv`, and `Pipeline_<...>.mq5`.

## Layout

- `PLAN.md` / `PROGRESS.md` — phase plan and live status
- `docs/MARKET_RULES.md` — the market-concept vocabulary (12 families) and
  exactly how each concept becomes a feature
- `pipeline/data/` — providers (Coinbase, Bitstamp, Kraken, HistData
  forex/metals, CSV/MT5 export) + window permutations
- `pipeline/features/` — ~100 causal features: market structure
  (HH/HL/LH/LL, BOS, CHoCH, MSS), trend regimes, SMC, classical patterns,
  candles, volume (incl. rolling volume profile), volatility, momentum
  divergences, S/R, sessions, options stub
- `pipeline/labeling.py` — triple-barrier labels
- `pipeline/model.py` — shadow-feature relevance + purged walk-forward LightGBM
- `pipeline/signal.py`, `pipeline/backtest.py` — CALL/PUT + TP/SL/RR,
  cost-aware evaluation with permutation p-value
- `pipeline/mql5/` — distillation to 12 portable features + EA template
- `tests/` — per-phase tests (`python -m pytest -q`)
- `external/` — cloned reference repos (gitignored; see
  `scripts/clone_references.sh`)
- `reports/` — committed summaries of real runs

## Honesty contract

Numbers you see in reports and EA headers are out-of-sample, net of costs,
with decision thresholds chosen on data the evaluation never sees. An AUC
near 0.5 / p-value above 0.05 means **no dependable edge was found** — the
pipeline says so rather than hiding it. Nothing here is financial advice;
markets can and do take the money of well-tested systems too.
