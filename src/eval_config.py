"""Shared evaluation grid + deterministic per-cell seeding, factored into
its own module with zero ML-framework imports (just numpy/pathlib) so
that `evaluate_gnn.py` (TensorFlow) and `evaluate_rl.py` (PyTorch) can
both depend on it without either one transitively importing the other's
framework, which is exactly the mistake that caused a segfault earlier
in this project (see `evaluate.py`'s docstring for the full story). Keep
it that way: don't import torch or tensorflow into this file.
"""
from __future__ import annotations

from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DISTANCES = (3, 5, 7, 9, 11)
P_VALUES = (0.03, 0.06, 0.1, 0.15, 0.2, 0.25)
N_SHOTS = 3000


def cell_seed(distance: int, p: float, base_seed: int = 123) -> int:
    """Deterministic per-cell seed. `evaluate_gnn.py` and `evaluate_rl.py`
    each call this independently (never sharing memory, running in
    separate processes) and must get the same seed for the same
    (distance, p) so they evaluate on identical shots."""
    return base_seed * 1_000_000 + distance * 10_000 + int(round(p * 1000))
