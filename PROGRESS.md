# Progress Tracker

| Phase | Status | Evidence |
|---|---|---|
| 0 — workspace, reference clones, reading rules | ✅ done | `docs/MARKET_RULES.md`, `scripts/clone_references.sh`, 8 repos in `external/` |
| 1 — data layer + window permutations | ✅ done | `tests/test_data.py` 6/6 passed (incl. live Coinbase + Bitstamp) |
| 2 — feature engine | ✅ done | `tests/test_features.py` 15/15 incl. no-lookahead proof |
| 3 — labeling + walk-forward model + relevance | ✅ done | `tests/test_model.py` 4/4 (planted signal kept & learned OOS) |
| 4 — signal engine (CALL/PUT, TP/SL, RR) + backtest | ✅ done | `tests/test_signal.py` 4/4 (informed beats random, costs verified) |
| 5 — MQL5 generator | ✅ done | `tests/test_mql5.py` 3/3; EA renders with measured distilled AUC |
| 6 — end-to-end on 2 exchanges + report | ✅ done | Coinbase BTC-USD + Bitstamp ETH-USD, `reports/RESEARCH_SUMMARY.md` |

| 7 — pattern benchmark ≥95%/pattern | ✅ done | `reports/PATTERN_BENCH.md`: mean 98.3%, min 95%, regression-tested |
| 8 — time-series models (AR + causal HMM) | ✅ done | `tests/test_timeseries.py` (incl. causality tests) |
| 9 — RL agent + recursive refit | ✅ done | `tests/test_rl.py`; integrated in `pipeline/run.py` steps 3b/4b |

Key findings from the two-exchange runs are in
`reports/RESEARCH_SUMMARY.md`: market structure dominates on both
instruments, candle patterns are near-worthless, the useful metric mix
shifts per window/instrument, and no cost-surviving edge was found on
crypto 1h — reported honestly in the generated EA headers.

## Notes & environment findings

* Yahoo Finance / stooq / Binance are **not reachable** from this
  environment (proxy/geo blocks). Verified working keyless sources:
  **Coinbase Exchange**, **Bitstamp**, **Kraken** (crypto), **histdata.com**
  (forex/metals M1), plus local CSV / MT5 exports. yfinance is therefore
  not a dependency.
* `twopirllc/pandas-ta` is no longer public and `pandas-ta` is gone from
  PyPI → cloned the `xgboosted/pandas-ta-classic` fork instead.
* ElliottWaveAnalyzer repo is gone → using the `taew` PyPI package.
* TA-Lib needs a system C library (not installable here) → candle patterns
  are implemented from candle geometry directly (see MARKET_RULES.md §8).
* Options-chain data (IV, greeks, OI) has no free keyless source → schema
  stubbed, auto-dropped by relevance stage (MARKET_RULES.md §12).
