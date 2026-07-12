"""Time-series model features (Phase 8).

Two classic time-series algorithms, both strictly causal:

* Rolling AR(p): OLS on the last `window` returns with p lags, refit every
  `refit` bars; the one-step-ahead forecast (in vol units) and its recent
  hit-rate become features. (cf. external/machine-learning-for-trading ch.9)
* Gaussian HMM regimes: fitted on PAST returns only (expanding refits every
  `refit_hmm` bars), then *forward-filtered* — the state probability at bar
  t uses observations up to t only. Smoothed posteriors would leak the
  future; we never use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_STATES = 3


def ar_forecast(close: pd.Series, p: int = 5, window: int = 400, refit: int = 20) -> pd.DataFrame:
    r = close.pct_change().fillna(0.0).values
    n = len(r)
    fc = np.full(n, np.nan)
    coefs = None
    for t in range(window + p, n):
        if coefs is None or (t - window - p) % refit == 0:
            seg = r[t - window : t]
            Y = seg[p:]
            X = np.column_stack([seg[p - j - 1 : len(seg) - j - 1] for j in range(p)])
            X = np.column_stack([np.ones(len(Y)), X])
            coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
        x = np.concatenate([[1.0], r[t - p : t][::-1]])
        fc[t] = float(x @ coefs)
    vol = pd.Series(r, index=close.index).rolling(200).std().replace(0, np.nan)
    fc_s = pd.Series(fc, index=close.index)
    out = pd.DataFrame(index=close.index)
    out["ts_ar_fc"] = (fc_s / vol).clip(-5, 5)          # forecast in vol units
    hit = (np.sign(fc_s.shift(1)) == np.sign(pd.Series(r, index=close.index))).astype(float)
    out["ts_ar_hitrate"] = hit.rolling(100).mean()      # is the AR model working lately?
    return out


def hmm_regimes(close: pd.Series, refit_hmm: int = 1000, min_train: int = 1500) -> pd.DataFrame:
    from hmmlearn.hmm import GaussianHMM

    r = close.pct_change().fillna(0.0).values.reshape(-1, 1)
    n = len(r)
    probs = np.full((n, N_STATES), np.nan)
    mean_state = np.full(n, np.nan)
    vol_state = np.full(n, np.nan)

    model = None
    order = None  # state indices sorted by mean return (stable meaning)
    log_pi = log_A = means = stds = None
    alpha = None  # forward filter state (log domain, normalised)

    def _obs_logpdf(x):
        return -0.5 * (((x - means) / stds) ** 2) - np.log(stds) - 0.5 * np.log(2 * np.pi)

    for t in range(n):
        if t >= min_train and (t == min_train or (t - min_train) % refit_hmm == 0):
            m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=50, random_state=0)
            try:
                m.fit(r[:t])
                model = m
                means = m.means_[:, 0].copy()
                stds = np.sqrt(m.covars_[:, 0, 0] if m.covars_.ndim == 3 else m.covars_[:, 0])
                order = np.argsort(means)  # 0=bear, 1=neutral, 2=bull
                log_pi = np.log(np.clip(m.startprob_, 1e-12, 1))
                log_A = np.log(np.clip(m.transmat_, 1e-12, 1))
                alpha = None  # restart filter with new params
            except Exception:
                pass
        if model is None:
            continue
        lp = _obs_logpdf(r[t, 0])
        if alpha is None:
            alpha = log_pi + lp
        else:
            alpha = lp + np.logaddexp.reduce(alpha[:, None] + log_A, axis=0)
        alpha = alpha - np.logaddexp.reduce(alpha)  # normalise (filtered posterior)
        post = np.exp(alpha)
        probs[t] = post[order]
        mean_state[t] = float(post @ means) / max(1e-12, np.abs(means).max())
        vol_state[t] = float(post @ stds) / max(1e-12, stds.mean())

    out = pd.DataFrame(index=close.index)
    out["ts_hmm_bear"] = probs[:, 0]
    out["ts_hmm_mid"] = probs[:, 1]
    out["ts_hmm_bull"] = probs[:, 2]
    out["ts_hmm_drift"] = np.clip(mean_state, -3, 3)
    out["ts_hmm_vol"] = np.clip(vol_state, 0, 5)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.concat([ar_forecast(close), hmm_regimes(close)], axis=1)
    return out
