# Progress Tracker

| Phase | Status | Evidence |
|---|---|---|
| 0 — workspace, reference clones, reading rules | ✅ done | `docs/MARKET_RULES.md`, `scripts/clone_references.sh`, 8 repos in `external/` |
| 1 — data layer + window permutations | ✅ done | `tests/test_data.py` 6/6 passed (incl. live Coinbase + Bitstamp) |
| 2 — feature engine | ⏳ in progress | |
| 3 — labeling + walk-forward model + relevance | ⏳ pending | |
| 4 — signal engine (CALL/PUT, TP/SL, RR) + backtest | ⏳ pending | |
| 5 — MQL5 generator | ⏳ pending | |
| 6 — end-to-end on 2 exchanges + report | ⏳ pending | |

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
