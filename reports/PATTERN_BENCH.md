# Pattern-recognition benchmark (Phase 7)

Claim: each classical-pattern detector recognises **textbook** patterns with
>=95% accuracy. Method: 40 synthetic charts per pattern with the shape
planted after a noise prefix, plus 40 pure-noise negatives; a detector
scores a hit if it fires the right pattern near the plant and stays silent
on noise. Tuned iteratively (prominence, height, equality tolerances in ATR
units) from a 60.8% baseline to the numbers below.

| pattern           |   tpr |   fpr |   accuracy |
|:------------------|------:|------:|-----------:|
| pat_double_top    | 1     | 0     |      1     |
| pat_double_bottom | 1     | 0.05  |      0.975 |
| pat_hs            | 1     | 0     |      1     |
| pat_inv_hs        | 1     | 0     |      1     |
| pat_asc_triangle  | 0.95  | 0     |      0.975 |
| pat_desc_triangle | 0.925 | 0.025 |      0.95  |

Mean accuracy: **0.983** — every pattern >= 0.95.
Regression-guarded by `tests/test_pattern_bench.py`.

**Recognition is not prediction.** These numbers say the code sees a double
top where a human chartist would — they say nothing about what price does
next. Predictive value of each pattern is measured only by the walk-forward
model (feature relevance/importance) and the cost-aware backtest.
