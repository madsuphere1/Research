"""Phase 8 tests: AR forecast learns a real AR process; HMM regimes are
causal (filtered, not smoothed) and identify a planted regime switch."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features.timeseries import ar_forecast, hmm_regimes


def _series(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="1h", tz="UTC")
    return pd.Series(vals, index=idx)


def test_ar_learns_ar_process():
    rng = np.random.default_rng(0)
    n = 3000
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.5 * r[i - 1] + 0.002 * rng.standard_normal()
    close = _series(100 * np.exp(np.cumsum(r)))
    fc = ar_forecast(close)["ts_ar_fc"]
    actual = close.pct_change().shift(-1)  # next-bar return
    both = pd.concat([fc, actual], axis=1).dropna()
    corr = both.corr().iloc[0, 1]
    assert corr > 0.2, f"AR forecast corr {corr}"


def test_ar_causal():
    rng = np.random.default_rng(1)
    close = _series(100 * np.exp(np.cumsum(0.003 * rng.standard_normal(2000))))
    full = ar_forecast(close)
    trunc = ar_forecast(close.iloc[:1500])
    pd.testing.assert_frame_equal(full.iloc[:1500], trunc)


def test_hmm_finds_regime_switch():
    rng = np.random.default_rng(2)
    n = 4000
    # bull regime first half (+drift), bear+high-vol second half
    r = np.concatenate([
        0.004 + 0.005 * rng.standard_normal(n // 2),
        -0.004 + 0.012 * rng.standard_normal(n // 2),
    ])
    close = _series(100 * np.exp(np.cumsum(r)))
    hm = hmm_regimes(close, refit_hmm=1000, min_train=1500)
    got = hm.dropna()
    assert len(got) > 2000
    # after the switch (with the filter warmed), bear prob should dominate bull
    late = got.iloc[-800:]
    assert late["ts_hmm_bear"].mean() > late["ts_hmm_bull"].mean()
    # and the filtered vol state must read higher in the high-vol half.
    # (No bull-vs-bear assertion on the early window: the causally fitted
    # model has only ever seen one regime there, so its state labels are
    # not yet meaningful — that is honest filtering, not a defect.)
    early = got[(got.index >= close.index[1600]) & (got.index <= close.index[1950])]
    assert late["ts_hmm_vol"].mean() > early["ts_hmm_vol"].mean()
    assert late["ts_hmm_drift"].mean() < early["ts_hmm_drift"].mean()
