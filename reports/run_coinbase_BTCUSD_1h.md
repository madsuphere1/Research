# Pipeline run: coinbase:BTC-USD 1h

Generated 2026-07-12 14:36 UTC. Data: 17510 bars
2024-07-12 .. 2026-07-12. Label: triple-barrier
(horizon 24 bars, SL 1.5 ATR, RR 1.5), costs 0.05 R/round-trip.

## Behaviour across window permutations
| window      |   bars |   wf_auc |   n_kept | kept_families                           |
|:------------|-------:|---------:|---------:|:----------------------------------------|
| lb6m_off0m  |   4339 |   0.5112 |       29 | mom,ms,ses,smc,sr,trend,vlt,vol         |
| lb6m_off6m  |   4411 |   0.5823 |       39 | mom,ms,ses,smc,sr,trend,vlt,vol         |
| lb6m_off12m |   4344 |   0.51   |       58 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |
| lb12m_off0m |   8750 |   0.536  |       58 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |
| lb12m_off6m |   8755 |   0.534  |       62 | mom,ms,pat,ses,smc,sr,trend,vlt,vol     |

## Full-window model
- Purged walk-forward AUC: **0.5000** (folds [0.534, 0.508, 0.487, 0.507, 0.464])
- Features kept by relevance: 63 / 143
- Family importance (gain): {'ms': np.int64(12866), 'vol': np.int64(8367), 'ses': np.int64(7154), 'trend': np.int64(6226), 'vlt': np.int64(5430), 'sr': np.int64(4465), 'smc': np.int64(3673), 'pat': np.int64(2185), 'mom': np.int64(1268), 'cdl': np.int64(124)}

## Held-out backtest (second half of OOS, net of costs)
{
  "n_trades": 478,
  "net_r_per_trade": -0.0652,
  "win_rate": 0.3954,
  "profit_factor": 0.89,
  "max_dd_r": 39.69,
  "p_value": 0.8982
}

Thresholds (chosen on first OOS half only): CALL >= 0.70, PUT <= 0.30.

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **0.5165**
- Spearman corr with full model OOS probabilities: 0.080
- EA: `Pipeline_coinbase_BTCUSD_1h.mq5` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
|                         |      0 |
|:------------------------|-------:|
| ses_asia_range_w        | 2471.5 |
| trend_meanrev           | 2080.1 |
| ms_ext_bars_since_choch | 2075.5 |
| sr_ema200_dist          | 1985.5 |
| vlt_atr_pct             | 1810.5 |
| vol_price_div           | 1717.3 |
| ms_ext_bars_since_bos   | 1658.7 |
| ses_dow                 | 1603.6 |
| vlt_rv_regime           | 1592.7 |
| ms_int_bars_since_choch | 1564.6 |
| vol_poc_dist            | 1562.5 |
| smc_ob_dist             | 1461.9 |
| vol_ad_slope            | 1450.9 |
| ms_ext_dist_swing_lo    | 1335.1 |
| ms_ext_dist_swing_hi    | 1305.9 |
| smc_bars_since_sweep    | 1261.1 |
| vlt_bb_width            | 1227.8 |
| vol_va_pos              | 1184.7 |
| vol_trend               | 1169.1 |
| ms_int_bars_since_bos   | 1083.2 |

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
