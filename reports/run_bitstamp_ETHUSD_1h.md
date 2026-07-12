# Pipeline run: bitstamp:ETH-USD 1h

Generated 2026-07-12 14:37 UTC. Data: 17521 bars
2024-07-12 .. 2026-07-12. Label: triple-barrier
(horizon 24 bars, SL 1.5 ATR, RR 1.5), costs 0.05 R/round-trip.

## Behaviour across window permutations
| window      |   bars |   wf_auc |   n_kept | kept_families                           |
|:------------|-------:|---------:|---------:|:----------------------------------------|
| lb6m_off0m  |   4344 |   0.4877 |       58 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |
| lb6m_off6m  |   4416 |   0.592  |       57 | mom,ms,pat,ses,smc,sr,trend,vlt,vol     |
| lb6m_off12m |   4344 |   0.544  |       59 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |
| lb12m_off0m |   8760 |   0.5672 |       75 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |
| lb12m_off6m |   8760 |   0.5309 |       61 | cdl,mom,ms,pat,ses,smc,sr,trend,vlt,vol |

## Full-window model
- Purged walk-forward AUC: **0.5253** (folds [0.524, 0.532, 0.482, 0.495, 0.594])
- Features kept by relevance: 73 / 143
- Family importance (gain): {'ms': np.int64(13060), 'vol': np.int64(8448), 'trend': np.int64(6913), 'ses': np.int64(6153), 'vlt': np.int64(4769), 'sr': np.int64(4634), 'smc': np.int64(3904), 'mom': np.int64(1644), 'pat': np.int64(1622), 'cdl': np.int64(510)}

## Held-out backtest (second half of OOS, net of costs)
{
  "n_trades": 558,
  "net_r_per_trade": -0.0252,
  "win_rate": 0.4158,
  "profit_factor": 0.956,
  "max_dd_r": 45.08,
  "p_value": 0.6906
}

Thresholds (chosen on first OOS half only): CALL >= 0.68, PUT <= 0.32.

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **0.5341**
- Spearman corr with full model OOS probabilities: 0.103
- EA: `Pipeline_bitstamp_ETHUSD_1h.mq5` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
|                         |      0 |
|:------------------------|-------:|
| ms_ext_bars_since_bos   | 2482   |
| ms_ext_bars_since_choch | 2070.5 |
| trend_meanrev           | 1953.4 |
| ses_asia_range_w        | 1792.6 |
| vol_trend               | 1760.8 |
| vlt_rv_regime           | 1652.7 |
| sr_ema200_dist          | 1597.8 |
| vol_price_div           | 1535.7 |
| vol_poc_dist            | 1518.5 |
| vol_va_pos              | 1459.8 |
| smc_ob_dist             | 1441.7 |
| trend_adx               | 1410.6 |
| vlt_atr_pct             | 1282   |
| trend_expansion         | 1189.5 |
| ses_dow                 | 1183.6 |
| ms_int_bars_since_bos   | 1181.7 |
| ses_month_cos           | 1172.4 |
| ms_int_bars_since_choch | 1120   |
| smc_bars_since_sweep    | 1102.1 |
| ses_month_sin           | 1071.6 |

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
