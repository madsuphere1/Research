"""Window-permutation engine.

The research question "does the behaviour hold on the last 6 months? the
last year? the year before that?" is answered by slicing one immutable
download into overlapping (lookback, offset) windows:

    lookbacks (months): e.g. 6, 12, 24
    offsets   (months back from the end of data): e.g. 0, 6, 12, 24

Each (lookback, offset) pair is a Window; the pipeline runs per-window and
compares metric relevance / model quality across windows to see whether the
instrument's behaviour is stable or regime-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    lookback_months: int
    offset_months: int  # 0 = ends at the most recent bar

    @property
    def name(self) -> str:
        return f"lb{self.lookback_months}m_off{self.offset_months}m"


def window_permutations(
    lookbacks=(6, 12, 24),
    offsets=(0, 6, 12),
    max_span_months: int | None = None,
) -> list[Window]:
    """All (lookback, offset) combinations that fit inside max_span_months."""
    out = []
    for lb in lookbacks:
        for off in offsets:
            if max_span_months is not None and lb + off > max_span_months:
                continue
            out.append(Window(lb, off))
    return out


def slice_window(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Slice a normalised OHLCV frame to one window (end-anchored)."""
    end = df.index.max() - pd.DateOffset(months=window.offset_months)
    start = end - pd.DateOffset(months=window.lookback_months)
    return df[(df.index > start) & (df.index <= end)]


def span_months(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return (df.index.max() - df.index.min()).days / 30.44
