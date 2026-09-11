"""Per-candidate-action feature vector for the RL decoder: the Q-network
scores a fixed-size feature vector per candidate qubit rather than
indexing a fixed output layer, which is what lets one trained model handle
any code distance (the action count, one per data qubit, changes with the
code).
"""
from __future__ import annotations

import numpy as np

from .rl_env import DecodingEnv

N_FEATURES = 6


def qubit_features(env: DecodingEnv, qubit_idx: int) -> np.ndarray:
    checks = env.graph.data_neighbor_checks[qubit_idx]
    degree = len(checks)
    active_count = sum(1 for c in checks for ci in [c - env.graph.n_data] if env.residual_syndrome[ci])

    # Net change in total residual weight this flip would cause: each
    # currently-inactive neighbour check would turn on (+1), each active
    # one would turn off (-1). Negative = strictly helps, 0 = pushes a
    # defect one step (still useful), positive = actively harmful.
    delta = degree - 2 * active_count

    # Distance (in qubit-index space) from this qubit to the nearest
    # currently-active check, as a coarse directional hint.
    active_check_positions = [c - env.graph.n_data + 0.5 for c in range(env.graph.n_checks)
                               if env.residual_syndrome[c]]
    if active_check_positions:
        nearest_dist = min(abs(qubit_idx - pos) for pos in active_check_positions)
    else:
        nearest_dist = 0.0

    total_active_frac = float(env.residual_syndrome.sum()) / max(1, env.graph.n_checks)

    return np.array([
        active_count / 2.0,
        (degree - 1),  # 0 for an endpoint qubit (degree 1), 1 for interior (degree 2)
        delta / 2.0,
        float(env.flipped[qubit_idx]),
        total_active_frac,
        nearest_dist / env.graph.n_data,
    ], dtype=np.float32)


def batch_features(env: DecodingEnv, candidates: list[int]) -> np.ndarray:
    return np.stack([qubit_features(env, q) for q in candidates]) if candidates else \
        np.zeros((0, N_FEATURES), dtype=np.float32)
