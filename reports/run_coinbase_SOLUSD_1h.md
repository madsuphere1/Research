# Pipeline run: coinbase:SOL-USD 1h

Generated 2026-07-12 20:17 UTC. Data: 17510 bars
2024-07-12 .. 2026-07-12. Label: triple-barrier
(horizon 24 bars, SL 1.5 ATR, RR 1.5), costs 0.05 R/round-trip.

## Behaviour across window permutations
| window       |   bars |   wf_auc |   n_kept | kept_families                              |
|:-------------|-------:|---------:|---------:|:-------------------------------------------|
| lb6m_off0m   |   4339 |   0.4942 |       70 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb6m_off6m   |   4411 |   0.5579 |       61 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb6m_off12m  |   4344 |   0.4976 |       53 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off0m  |   8750 |   0.543  |       72 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off6m  |   8755 |   0.4949 |       69 | cdl,mom,ms,ses,smc,sr,trend,ts,vlt,vol     |
| lb12m_off12m |   8760 |   0.5266 |       57 | mom,ms,pat,ses,smc,sr,trend,ts,vlt,vol     |
| lb24m_off0m  |  17510 |   0.53   |       83 | cdl,mom,ms,pat,ses,smc,sr,trend,ts,vlt,vol |

## Full-window model
- Purged walk-forward AUC: **0.5236** (folds [0.523, 0.54, 0.485, 0.505, 0.565])
- Features kept by relevance: 83 / 150
- Family importance (gain): {'ms': 12802, 'vol': 7101, 'trend': 6831, 'ses': 5609, 'ts': 4872, 'sr': 4298, 'vlt': 4002, 'smc': 3974, 'mom': 1706, 'cdl': 666, 'pat': 269}

## Held-out backtest (second half of OOS, net of costs)
{
  "n_trades": 486,
  "net_r_per_trade": 0.0544,
  "win_rate": 0.4424,
  "profit_factor": 1.098,
  "max_dd_r": 17.28,
  "p_value": 0.1218
}

Thresholds (chosen on first OOS half only): CALL >= 0.70, PUT <= 0.30.

## Recursive error-boosted refit (validation-early-stopped)
Validation AUC per round: [0.519, 0.5059, 0.5331, 0.5082, 0.5163] — kept round
2. Recursion stops when validation stops improving; training
error is never the stop criterion (that would just memorise the past).

## RL agent (contextual Q-learning, reward = net R, punishment = losses)
State = (model-probability bin, trend regime, volatility bin); actions
CALL/PUT/FLAT; trained by replaying past folds until the Q-table is stable
(fold epochs: ['200ep', '198ep*', '200ep*', '198ep*'], * = converged), then evaluated on the same held-out
half as the threshold policy:
{
  "n_trades": 708,
  "net_r_per_trade": 0.0054,
  "win_rate": 0.4237,
  "profit_factor": 1.009,
  "max_dd_r": 29.51,
  "p_value": 0.4591
}

## MQL5 export
- Distilled logistic (12 portable features) walk-forward AUC: **0.5141**
- Spearman corr with full model OOS probabilities: 0.044
- EA: `Pipeline_coinbase_SOLUSD_1h.mq5` (compile in MetaEditor; defaults embed the learned parameters)

## Top features (walk-forward gain)
|                         |      0 |
|:------------------------|-------:|
| ms_ext_bars_since_choch | 2704.7 |
| trend_meanrev           | 2190.6 |
| ms_ext_bars_since_bos   | 2056.1 |
| ses_asia_range_w        | 1914.7 |
| sr_ema200_dist          | 1761.5 |
| vol_trend               | 1647.7 |
| smc_ob_dist             | 1579.6 |
| vlt_rv_regime           | 1563.9 |
| vol_price_div           | 1467.8 |
| trend_adx               | 1432.6 |
| ms_int_bars_since_bos   | 1305.5 |
| vol_poc_dist            | 1296.7 |
| ts_ar_hitrate           | 1279.8 |
| vol_va_pos              | 1205.3 |
| ses_month_sin           | 1118.2 |
| vlt_atr_pct             | 1098.4 |
| trend_expansion         | 1020.2 |
| ms_int_bars_since_choch |  943.9 |
| ses_dow                 |  943.5 |
| smc_bars_since_sweep    |  929.5 |

## Read this honestly
A walk-forward AUC near 0.5 and/or a backtest p-value above 0.05 means NO
dependable edge was found for this instrument/timeframe under these costs —
the EA header carries the same numbers. Prior related work
(../Claude-researcg) found signals of this size do not survive retail costs.
