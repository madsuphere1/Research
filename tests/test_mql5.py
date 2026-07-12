"""Phase 5 tests: portable features are computable and causal, distillation
learns a planted signal, and the rendered EA has no leftover placeholders,
balanced braces, and embeds the coefficients."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.labeling import triple_barrier
from pipeline.mql5.generator import distill, render_ea
from pipeline.mql5.portable import PORTABLE_ORDER, compute_portable


def synth(n=2500, seed=5):
    rng = np.random.default_rng(seed)
    # momentum-driven series so the portable features carry signal
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = 0.25 * ret[i - 1] + 0.004 * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.roll(close, 1), "high": close * 1.003,
                       "low": close * 0.997, "close": close, "volume": 1.0}, index=idx)
    df.iloc[0, 0] = close[0]
    return df


DF = synth()


def test_portable_features_causal_and_bounded():
    P = compute_portable(DF)
    assert list(P.columns) == PORTABLE_ORDER
    assert P.iloc[250:].notna().all().all()
    assert (P.iloc[250:].abs() <= 50).all().all()
    # causality: truncation must not change earlier values
    P2 = compute_portable(DF.iloc[:1500])
    pd.testing.assert_frame_equal(P.iloc[:1500], P2)


def test_distill_learns_momentum():
    lab = triple_barrier(DF, horizon=12, sl_atr=1.5, rr=1.5)
    dm = distill(DF, lab, full_proba=None)
    assert dm.wf_auc > 0.52, dm.wf_auc
    assert len(dm.coefs) == len(PORTABLE_ORDER)


def test_render_ea_complete():
    lab = triple_barrier(DF, horizon=12, sl_atr=1.5, rr=1.5)
    dm = distill(DF, lab)
    code = render_ea(
        dm, symbol="BTC-USD", provider="coinbase", timeframe="1h",
        call_th=0.58, put_th=0.42, sl_atr=1.5, rr=1.5, horizon=12,
        full_auc=0.55, bt_summary={"n_trades": 100, "net_r_per_trade": 0.05},
    )
    assert not re.search(r"\{[A-Z_]+\}", code), "unfilled placeholder"
    assert code.count("{") == code.count("}")
    assert "OnTick" in code and "CTrade" in code
    assert f"{dm.intercept:.10g}" in code
    # all 12 coefficients present
    coefs_line = re.search(r"COEF\[NFEAT\] = \{ (.*?) \};", code).group(1)
    assert len(coefs_line.split(",")) == 12
    assert "CallThreshold = 0.58" in code
