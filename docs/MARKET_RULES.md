# Market Reading Rules — terms, definitions, and how the pipeline encodes them

This is the pipeline's "terms and conditions" of the market: the concept
vocabulary it reads price with, and the exact feature(s) each concept maps
to in `pipeline/features/`. Not every concept helps every instrument; the
feature-relevance stage (Phase 3) measures which ones carry signal for the
instrument being studied and drops the rest.

Reference implementations studied: `external/smart-money-concepts`,
`external/pandas-ta-classic`, `external/TradingPatternScanner`,
`external/pyharmonics`, `external/py-market-profile`,
`external/machine-learning-for-trading`,
`external/Financial-Models-Numerical-Methods`, `external/awesome-quant`.
(The original ElliottWaveAnalyzer repo is no longer public; the `taew`
PyPI library is used for Elliott-wave counts instead. TA-Lib requires a C
library not available here; `pandas-ta-classic` + our own vectorised
implementations cover the same ground.)

## 1. Market structure (most important)

| Term | Definition | Feature |
|---|---|---|
| Swing High / Swing Low | Local extreme confirmed by `k` bars on each side | `market_structure.swings` |
| Higher High (HH) | Swing high above previous swing high | `ms_hh` |
| Higher Low (HL) | Swing low above previous swing low | `ms_hl` |
| Lower High (LH) | Swing high below previous swing high | `ms_lh` |
| Lower Low (LL) | Swing low below previous swing low | `ms_ll` |
| Break of Structure (BOS) | Close beyond the last swing extreme **in trend direction** (continuation) | `ms_bos_dir`, `ms_bars_since_bos` |
| Change of Character (CHoCH) | Close beyond the last swing extreme **against trend** (first reversal warning) | `ms_choch_dir`, `ms_bars_since_choch` |
| Market Structure Shift (MSS) | CHoCH confirmed by an opposite HH/HL or LH/LL sequence | `ms_shift` |
| Internal structure | Structure of minor swings (small `k`) inside the external legs | `ms_int_*` (k=3) |
| External structure | Structure of major swings (large `k`) | `ms_ext_*` (k=8) |

## 2. Trend types

Encoded as a regime classifier over EMA slope, ADX, realised-vol and
efficiency ratio: strong/weak up, strong/weak down, sideways, expansion,
compression, acceleration, exhaustion, parabolic, mean-reverting.
Features: `trend_regime`, `trend_strength`, `trend_efficiency`,
`trend_accel`, `trend_parabolic`, `trend_meanrev` (`features/trend.py`).

## 3. Swing patterns (rhythm)

Impulse vs pullback/retracement depth (fib fraction of last leg), deep vs
shallow pullback, V-top/V-bottom sharpness, rounded turn curvature.
Features: `swing_leg_dir`, `swing_pullback_depth`, `swing_v_sharpness`
(`features/market_structure.py`).

## 4. Classical chart patterns

Head & Shoulders (+inverse), double/triple top/bottom, triangles, wedges,
channels, flags/pennants — detected from the swing sequence
(`features/classical.py`, approach cross-checked against
`external/TradingPatternScanner`). Features: `pat_*` one-hots + bars-since.

## 5. Smart Money Concepts (SMC / ICT)

| Term | Definition | Feature |
|---|---|---|
| Fair Value Gap (FVG) | 3-bar gap: bar1.high < bar3.low (bull) or bar1.low > bar3.high (bear) | `smc_fvg_dir`, `smc_fvg_dist` |
| Order Block | Last opposite candle before a displacement that breaks structure | `smc_ob_dir`, `smc_ob_dist` |
| Breaker / Mitigation block | Order block violated then retested | `smc_breaker` |
| Liquidity sweep / grab | Wick beyond an equal-highs/lows pool that closes back inside | `smc_sweep_dir`, `smc_bars_since_sweep` |
| Equal Highs / Equal Lows | ≥2 swing extremes within tolerance (resting liquidity) | `smc_eqh`, `smc_eql` |
| Premium / Discount | Position of price in the external swing range (>50% premium) | `smc_range_pos` |
| Balanced Price Range | Overlapping opposing FVGs | via `smc_fvg_*` |
| Inducement | Minor liquidity pool just before a POI | proxied by internal sweep `smc_int_sweep` |
| Optimal Trade Entry (OTE) | 0.62–0.79 retracement of last impulse | `smc_ote_zone` |

Cross-checked against `external/smart-money-concepts` (also installed as
the `smartmoneyconcepts` package and used directly where its API fits).

## 6. Volume patterns

Volume spike/climax (z-score), dry-up, volume divergence vs price,
participation trend, accumulation/distribution (A/D line + OBV slopes),
volume-profile position (POC/value-area distance, after
`external/py-market-profile`). Features: `vol_*` (`features/volume.py`).

## 7. Volatility patterns

ATR expansion/contraction ratio, Bollinger bandwidth squeeze (percentile)
and expansion, squeeze-release direction, realised-vol regime.
Features: `vlt_*` (`features/volatility.py`).

## 8. Candlestick patterns — geometry first

Raw geometry (body size, upper/lower wick ratios, close position in range,
gap) plus the classical named patterns (hammer, shooting star, doji,
engulfing, harami, morning/evening star, three soldiers/crows, marubozu,
spinning top) computed from that geometry. Features: `cdl_*`
(`features/candles.py`).

## 9. Momentum patterns

RSI / MACD regular and hidden divergence (vs swing points), momentum
acceleration and exhaustion. Features: `mom_*` (`features/momentum.py`).

## 10. Support & Resistance

Horizontal S/R from swing clustering, dynamic S/R (EMA distances), flip
zones, pivot points, supply/demand zone distances. Features: `sr_*`
(`features/support_resistance.py`).

## 11. Session & time features

Hour-of-day (sin/cos), day-of-week, month, Asian range width/position,
London and New York opens/kill-zones, overlap session flag — computed in
the instrument's quoted timezone (UTC by default). Features: `ses_*`
(`features/sessions.py`). Only meaningful for intraday timeframes; the
relevance stage drops them on daily data automatically.

## 12. Options-specific features

IV/greeks/OI/put-call/max-pain/GEX require an options-chain feed, which no
free keyless source provides. The schema is defined in
`features/options_stub.py` so a chain provider can be plugged in later;
these features stay NaN (and are auto-dropped) when no chain is supplied.
"CALL/PUT" in this pipeline's output is a **directional** decision (also
usable for spot/CFD long/short); option pricing itself is out of scope
until a chain source exists — see `external/Financial-Models-Numerical-Methods`
for the pricing models to use then.

## Cost & honesty rules (inherited from ../Claude-researcg)

* Every backtest number is reported **net of spread/fee**.
* Walk-forward only; purge a gap between train and test.
* A metric family that doesn't beat permuted-importance noise for the
  instrument is excluded from the final model and the generated EA.
