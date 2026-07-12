"""Phase 7 regression test: every classical-pattern detector must hold
>=95% accuracy on the labeled textbook benchmark (recognition, NOT
prediction — see pipeline/bench/pattern_bench.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.bench import run_benchmark


def test_every_pattern_at_least_95():
    df = run_benchmark(n_per_class=40)
    failing = df[df.accuracy < 0.95]
    assert failing.empty, f"below 95%:\n{failing}"
    assert (df.tpr >= 0.9).all()
    assert (df.fpr <= 0.05).all()
