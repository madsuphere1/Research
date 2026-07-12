# Pipeline run: bitstamp:ETH-USD 1h

Generated 2026-07-12 15:03 UTC. Data: 17521 bars
2024-07-12 .. 2026-07-12. Label: triple-barrier
(horizon 24 bars, SL 1.5 ATR, RR 1.5), costs 0.05 R/round-trip.

## Behaviour across window permutations
| window       |   bars |   wf_auc |   n_kept | kept_families                          |
|:-------------|-------:|---------:|---------:|:---------------------------------------|
| lb6m_off0m   |   4344 |   0.497  |       48 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb6m_off6m   |   4416 |   0.5791 |       53 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb6m_off12m  |   4344 |   0.5521 |       60 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb12m_off0m  |   8760 |   0.5692 |       66 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb12m_off6m  |   8760 |   0.5431 |       73 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb12m_off12m |   8760 |   0.5206 |       61 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |
| lb24m_off0m  |  17520 |   0.5437 |       80 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol |

## Full-window model
- Purged walk-forward AUC: **0.5473** (folds [0.532, 0.556, 0.506, 0.57, 0.573])
- Features kept by relevance: 82 / 150
- Family importance (gain): {'ms': 12500, 'vol': 8124, 'trend': 6688, 'ses': 5468, 'ts': 4974, 'vlt': 4599, 'sr': 4269, 'smc': 3795, 'mom': 1954, 'cdl': 491}

## Held-out backtest (second half of OOS, net of costs)
{
  "n_trades": 507,
  "net_r_per_trade": -0.0151,
  "win_rate": 0.4201,
  "profit_factor": 0.974,
  "max_dd_r": 27.34,
  "p_value": 0.6427
}

Thresholds (chosen on first OOS half only): CALL >= 0.70, PUT <= 0.30.

## Recursive error-boosted refit (validation-early-stopped)
Validation AUC per round: [0.5695, 0.561, 0.5488] — kept round
0. Recursion stops when validation stops improving; training
error is never the stop criterion (that would just memorise the past).

## RL agent (contextual Q-learning, reward = net R, punishment = losses)
State = (model-probability bin, trend regime, volatility bin); actions
CALL/PUT/FLAT; trained by replaying past folds until the Q-table is stable
(fold epochs: ['198ep*', '200ep*', '200ep', '198ep*'], * = converged), then evaluated on the same held-out
half as the threshold policy:
{
  "n_trades": 704,
  "net_r_per_trade": -0.0039,
  "win_rate": 0.4276,
  "profit_factor": 0.993,
  "max_dd_r": 31.65,
  "p_value": 0.5629
}

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **0.5341**
- Spearman corr with full model OOS probabilities: 0.114
- EA: `Pipeline_bitstamp_ETHUSD_1h.mq5` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
|                         |      0 |
|:------------------------|-------:|
| ms_ext_bars_since_bos   | 2209   |
| trend_meanrev           | 1904.4 |
| vol_trend               | 1743.7 |
| ses_asia_range_w        | 1671.5 |
| vlt_rv_regime           | 1580   |
| ms_ext_bars_since_choch | 1569.4 |
| sr_ema200_dist          | 1496.4 |
| vol_price_div           | 1493.5 |
| vol_poc_dist            | 1484.5 |
| smc_ob_dist             | 1449   |
| trend_adx               | 1398.8 |
| ts_ar_hitrate           | 1358.1 |
| vlt_atr_pct             | 1325.8 |
| vol_va_pos              | 1273.7 |
| ms_int_bars_since_bos   | 1167.7 |
| ms_int_bars_since_choch | 1108.5 |
| ms_ext_dist_swing_lo    | 1099.5 |
| ses_dow                 | 1070   |
| trend_expansion         | 1054.4 |
| ms_ext_dist_swing_hi    | 1013.9 |

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
