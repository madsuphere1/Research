# Agent guide (CLAUDE.md)

Behaviour-driven market prediction pipeline. Read `PLAN.md` (phases),
`PROGRESS.md` (status), `docs/MARKET_RULES.md` (concept vocabulary),
`TESTING.md` (test protocol + ledger) before making changes.

## Hard rules from the user — do not violate

1. **Never bypass computation steps.** A feature-matrix cache was added
   once and the user asked for it to be reverted (commit `ae03fb6`,
   reverted). Every run must compute data → features → model fresh.
   Do not re-add caching layers, shortcuts, or "fast paths".
2. **Run one exchange/instrument at a time.** No parallel simulations —
   they contend for CPU and the user prefers sequential, observable runs.
3. **Every user test is logged.** When the user phrases a scenario
   ("if I had X in Y over Z at L leverage", "which timeframe", "what does
   live data fit"), run the matching CLI with `--test-name testN_...` and
   `--prompt "<their words verbatim>"` so it lands in
   `logs/test_runs.jsonl`. Then commit and push the logs.
4. **Honesty is the product.** Every reported number must be
   out-of-sample, cost-adjusted, with thresholds/policies fitted on data
   the evaluation never sees. If a result is not statistically significant
   or a sample is small, say so in the reply. Never present in-sample
   results, never fit anything on evaluation data, never soften a losing
   result. "Recursion until true" stops on validation, never on test.

## Environment facts (verified in this container)

* Reachable keyless data: **Coinbase Exchange, Bitstamp, Kraken** (crypto),
  **histdata.com** (forex/metals M1; archives lag ~2 weeks). Yahoo, stooq,
  Binance are blocked. Stocks/options need a CSV (MT5 export supported) —
  there is no keyless stock feed here.
* `pandas-ta` is gone from PyPI (use `external/pandas-ta-classic` clone);
  ElliottWaveAnalyzer repo is gone (`taew` package instead); TA-Lib's C
  lib can't be installed (candle patterns are implemented from geometry).
* Third-party reference repos live in `external/` (gitignored);
  re-clone with `scripts/clone_references.sh`.
* Git: `user.email noreply@anthropic.com`, `user.name Claude`; branch
  `claude/trading-ml-market-structure-4fhn18` in this repo AND the sister
  repo `../Claude-researcg`. Push with `git push -u origin <branch>`.

## Commands

```bash
python -m pytest -q                      # full suite (tests/ only; ~50 tests)
python -m pipeline.run      --provider coinbase --symbol BTC-USD --timeframe 1h --years 2
python -m pipeline.simulate --provider coinbase --symbol SOL-USD --balance 6000 \
        --leverage 1 --days 61 [--start YYYY-MM-DD --end YYYY-MM-DD] \
        --test-name testN --prompt "..."
python -m pipeline.scan     --provider histdata --symbol XAUUSD --test-name testN --prompt "..."
python -m pipeline.match    --provider coinbase --symbol BTC-USD --timeframe 1h \
        --test-name testN --prompt "..."
```

`pipeline.run` = research report + MQL5 EA (outputs/<tag>/, copy keepers to
reports/). `simulate` = dollar replay of OOS decisions. `scan` = timeframe
ladder (15m/1h/6h/1d). `match` = live regime-conditional pattern snapshot.

## Architecture in one paragraph

`pipeline/data/` providers normalise everything to UTC OHLCV; `features/`
builds ~150 causal features (market structure, trend, SMC, classical
patterns, candles, volume, volatility, momentum, S/R, sessions,
AR/HMM time-series, options stub); `labeling.py` triple-barrier;
`model.py` shadow-feature relevance + purged walk-forward LightGBM;
`recursive.py` error-boosted refit (validation early stop); `rl.py`
contextual Q-agent (reward = realised net R); `signal*.py` CALL/PUT with
ATR TP/SL at fixed RR; `backtest.py`/`simulate.py` cost-aware evaluation;
`mql5/` distils to 12 portable features and renders a compilable EA;
`bench/` pattern-recognition benchmark (every detector must stay >=95% —
regression-tested).

## Invariants the tests enforce (do not break)

* No lookahead: features at bar t identical with/without future bars
  (`tests/test_features.py::test_no_lookahead`).
* Swings/patterns only count once confirmed k bars later.
* Every classical-pattern detector >=95% on the labeled benchmark
  (`tests/test_pattern_bench.py`).
* Backtest: next-bar-open entry, SL wins intra-bar ties, one position at
  a time, costs subtracted per round trip.
* Purge gap >= label horizon between train and test folds.

## Current research state (2026-07-12)

Best unproven lead: **BTC-USD on 6h candles** (+0.177 R/trade held-out,
p=0.024 raw, ~0.10 after timeframe-selection correction) — needs
confirmation on a fresh instrument/period before being believed. The
system consistently beats buy-and-hold in falling markets and lags strong
bull trends; policies are regime-dependent (see TESTING.md ledger).

**Live correction policy (tests 8–10):** in-window rolling retraining +
circuit breaker are default-on in `pipeline/simulate.py`. RL over adaptive
probabilities improved every instrument tried. Threshold cutoffs must be
calibrated on the same probability source they filter — the simulator runs
`threshold_static`, `threshold_adaptive_recalibrated` and `rl_adaptive`
side by side; never mix a static calibration with adaptive probabilities
(test 9 showed that flips a +21.7% window to −19.0%).
