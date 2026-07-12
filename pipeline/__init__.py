"""Behaviour-driven market prediction pipeline.

data -> windows -> features -> relevance -> walk-forward model -> signal
(CALL/PUT + TP/SL/RR) -> backtest -> MQL5 EA generation.
"""

__version__ = "0.1.0"
