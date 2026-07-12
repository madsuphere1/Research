# Pipeline run: coinbase:BTC-USD 1h

Generated 2026-07-12 15:00 UTC. Data: 17510 bars
2024-07-12 .. 2026-07-12. Label: triple-barrier
(horizon 24 bars, SL 1.5 ATR, RR 1.5), costs 0.05 R/round-trip.

## Behaviour across window permutations
| window       |   bars |   wf_auc |   n_kept | kept_families                              |
|:-------------|-------:|---------:|---------:|:-------------------------------------------|
| lb6m_off0m   |   4339 |   0.5158 |       54 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb6m_off6m   |   4411 |   0.594  |       37 | mom,ms,ses,smc,sr,trend,ts,vlt,vol         |
| lb6m_off12m  |   4344 |   0.4752 |       38 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off0m  |   8750 |   0.5383 |       61 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off6m  |   8755 |   0.5278 |       74 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off12m |   8760 |   0.5406 |       74 | cdl,mom,ms,pat,ses,smc,sr,trend,ts,vlt,vol |
| lb24m_off0m  |  17510 |   0.5083 |       88 | cdl,mom,ms,pat,ses,smc,sr,trend,ts,vlt,vol |

## Full-window model
- Purged walk-forward AUC: **0.5092** (folds [0.515, 0.524, 0.504, 0.511, 0.493])
- Features kept by relevance: 88 / 150
- Family importance (gain): {'ms': 12007, 'vol': 7691, 'ses': 5994, 'trend': 5859, 'vlt': 5081, 'ts': 4930, 'sr': 3995, 'smc': 3586, 'mom': 1665, 'pat': 528, 'cdl': 527}

## Held-out backtest (second half of OOS, net of costs)
{
  "n_trades": 470,
  "net_r_per_trade": -0.047,
  "win_rate": 0.3957,
  "profit_factor": 0.92,
  "max_dd_r": 35.58,
  "p_value": 0.8483
}

Thresholds (chosen on first OOS half only): CALL >= 0.70, PUT <= 0.30.

## Recursive error-boosted refit (validation-early-stopped)
Validation AUC per round: [0.4574, 0.4737, 0.4698, 0.4774, 0.4779, 0.4758, 0.4734] — kept round
4. Recursion stops when validation stops improving; training
error is never the stop criterion (that would just memorise the past).

## RL agent (contextual Q-learning, reward = net R, punishment = losses)
State = (model-probability bin, trend regime, volatility bin); actions
CALL/PUT/FLAT; trained by replaying past folds until the Q-table is stable
(fold epochs: ['198ep*', '198ep*', '198ep*', '193ep*'], * = converged), then evaluated on the same held-out
half as the threshold policy:
{
  "n_trades": 683,
  "net_r_per_trade": 0.0252,
  "win_rate": 0.4319,
  "profit_factor": 1.045,
  "max_dd_r": 26.79,
  "p_value": 0.2794
}

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **0.5165**
- Spearman corr with full model OOS probabilities: 0.107
- EA: `Pipeline_coinbase_BTCUSD_1h.mq5` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
|                         |      0 |
|:------------------------|-------:|
| ses_asia_range_w        | 2152.5 |
| ms_ext_bars_since_choch | 1732.5 |
| trend_meanrev           | 1699.7 |
| sr_ema200_dist          | 1676   |
| vlt_atr_pct             | 1551.1 |
| vol_ad_slope            | 1516.3 |
| ms_ext_bars_since_bos   | 1512.1 |
| vol_price_div           | 1496.5 |
| ses_dow                 | 1463.7 |
| vol_poc_dist            | 1434.6 |
| vlt_rv_regime           | 1428.5 |
| ms_int_bars_since_choch | 1421   |
| smc_ob_dist             | 1398.4 |
| ms_ext_dist_swing_hi    | 1172.1 |
| vol_trend               | 1118.5 |
| ts_hmm_drift            | 1079.8 |
| ts_ar_hitrate           | 1071.9 |
| ms_ext_dist_swing_lo    | 1063.7 |
| ms_int_bars_since_bos   | 1049.1 |
| vlt_bb_width            | 1028.4 |

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
