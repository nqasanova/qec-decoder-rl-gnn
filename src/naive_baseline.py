"""The weakest sensible baseline: a purely local decoder that flips a data
qubit whenever it has strictly more active neighbouring checks than
inactive ones, breaking ties by not flipping. Unlike `exact_map_decoder`
this has no idea it needs to eventually reach *one specific* global
correction; it just reacts to local syndrome pressure, so it's a fair
stand-in for "the simplest thing you might try before building a real
decoder."
"""
from __future__ import annotations

import numpy as np


def naive_local_decoder(syndrome: np.ndarray) -> np.ndarray:
    """syndrome: (n_shots, n_checks) -> correction: (n_shots, n_data).
    Data qubit i has checks i-1 and i as neighbours (missing at the
    boundary); flip it iff more than half of its existing neighbour checks
    are active.
    """
    n_shots, n_checks = syndrome.shape
    n_data = n_checks + 1
    correction = np.zeros((n_shots, n_data), dtype=np.uint8)
    for i in range(n_data):
        neighbor_checks = [c for c in (i - 1, i) if 0 <= c < n_checks]
        active = sum(syndrome[:, c] for c in neighbor_checks)
        correction[:, i] = (active > len(neighbor_checks) / 2).astype(np.uint8)
    return correction
