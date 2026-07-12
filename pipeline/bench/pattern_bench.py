"""Pattern-recognition benchmark.

Claim under test: the classical-pattern detectors recognise *textbook*
patterns with >=95% accuracy. We generate labeled synthetic charts — a
noisy base series with a clean planted pattern (or none, for negatives) —
and score each detector:

    accuracy = (fires on planted pattern + stays silent on negatives) / all

Detection accuracy is NOT prediction accuracy: recognising a double top at
95% says nothing about whether price falls afterwards; that question is
answered only by the walk-forward model/backtest. This file keeps the two
claims separate on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import classical

BENCH_PATTERNS = [
    "pat_double_top", "pat_double_bottom", "pat_hs", "pat_inv_hs",
    "pat_asc_triangle", "pat_desc_triangle",
]

_SWING_W = 22  # bars per synthetic swing leg (>= 2k+1 so k=5 swings confirm)


def _leg(a: float, b: float, n: int = _SWING_W) -> np.ndarray:
    """Smooth leg from a to b with a rounded turn (half-cosine)."""
    t = np.linspace(0, np.pi, n)
    return a + (b - a) * (1 - np.cos(t)) / 2


def _shape(kind: str, base: float, amp: float) -> np.ndarray:
    """Piecewise-linear textbook pattern, as a sequence of swing targets."""
    b, A = base, amp
    if kind == "pat_double_top":
        pts = [b, b + A, b + 0.45 * A, b + 1.02 * A, b + 0.4 * A]
    elif kind == "pat_double_bottom":
        pts = [b, b - A, b - 0.45 * A, b - 1.02 * A, b - 0.4 * A]
    elif kind == "pat_hs":
        pts = [b, b + 0.75 * A, b + 0.3 * A, b + 1.15 * A, b + 0.3 * A, b + 0.78 * A, b + 0.25 * A]
    elif kind == "pat_inv_hs":
        pts = [b, b - 0.75 * A, b - 0.3 * A, b - 1.15 * A, b - 0.3 * A, b - 0.78 * A, b - 0.25 * A]
    elif kind == "pat_asc_triangle":
        pts = [b, b + A, b + 0.25 * A, b + 1.02 * A, b + 0.55 * A, b + 1.03 * A, b + 0.75 * A]
    elif kind == "pat_desc_triangle":
        pts = [b, b - A, b - 0.25 * A, b - 1.02 * A, b - 0.55 * A, b - 1.03 * A, b - 0.75 * A]
    else:
        raise ValueError(kind)
    return np.concatenate([_leg(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])


def generate_labeled_chart(kind: str | None, seed: int, n: int = 420, noise: float = 0.06):
    """OHLC frame with `kind` planted at the end (or pure noise if None)."""
    rng = np.random.default_rng(seed)
    base = 100.0
    amp = 8.0
    drift = rng.standard_normal(n).cumsum() * noise * 0.6
    close = base + drift
    if kind is not None:
        shape = _shape(kind, close[-1], amp)
        m = len(shape)
        close = np.concatenate([close, shape + rng.standard_normal(m) * noise])
    idx = pd.date_range("2025-01-01", periods=len(close), freq="1h", tz="UTC")
    wig = np.abs(rng.standard_normal(len(close))) * noise * 2
    df = pd.DataFrame(
        {"open": np.roll(close, 1), "high": close + wig, "low": close - wig,
         "close": close, "volume": 1000.0},
        index=idx,
    )
    df.iloc[0, 0] = close[0]
    return df


def detect(df: pd.DataFrame, kind: str, tail: int = 60) -> bool:
    """Did the detector fire the right pattern near the end of the chart?"""
    feats = classical.compute(df)
    return bool((feats[kind].iloc[-tail:] != 0).any())


def run_benchmark(n_per_class: int = 40, noise: float = 0.06, seed0: int = 0) -> pd.DataFrame:
    """Per-pattern true-positive rate, false-positive rate on negatives, accuracy."""
    rows = []
    # negatives once, shared across detectors
    negs = [generate_labeled_chart(None, seed0 + 10_000 + i, noise=noise) for i in range(n_per_class)]
    for kind in BENCH_PATTERNS:
        tp = sum(detect(generate_labeled_chart(kind, seed0 + i, noise=noise), kind)
                 for i in range(n_per_class))
        fp = sum(detect(df, kind) for df in negs)
        acc = (tp + (n_per_class - fp)) / (2 * n_per_class)
        rows.append({"pattern": kind, "tpr": tp / n_per_class,
                     "fpr": fp / n_per_class, "accuracy": acc})
    return pd.DataFrame(rows)
