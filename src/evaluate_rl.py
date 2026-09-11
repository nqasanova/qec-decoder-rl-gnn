"""Evaluate the RL decoder across the full (distance, p) grid, and write
results to data/processed/eval_rl_results.json.

Run as a **separate process** from `evaluate_gnn.py`; see that module's
docstring for why (a real TensorFlow/PyTorch segfault, not a
hypothetical). Uses the same `cell_seed()` derivation so the error shots
sampled here exactly match the ones `evaluate_gnn.py` used for the same
(distance, p) cell.
"""
from __future__ import annotations

import json

import numpy as np

from .dqn_agent import DQNAgent
from .eval_config import DISTANCES, N_SHOTS, P_VALUES, PROCESSED_DIR, cell_seed
from .repetition_code import RepetitionCode, majority_vote_logical_readout, sample_errors
from .tanner_graph import build_tanner_graph
from .train_rl import RL_MODEL_PATH, run_episode

OUT_PATH = PROCESSED_DIR / "eval_rl_results.json"
TRAIN_DISTANCES_RL = (5, 7, 9, 11)  # what train_rl.train()'s make_random_training_instance samples


def rl_decode(agent: DQNAgent, distance: int, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The RL agent decodes one shot at a time (it's a sequential
    decision process, unlike the GNN/exact decoders); returns
    (correction, completed) for the whole batch."""
    graph = build_tanner_graph(distance)
    n_shots = errors.shape[0]
    correction = np.zeros((n_shots, distance), dtype=np.uint8)
    completed = np.zeros(n_shots, dtype=bool)
    for i in range(n_shots):
        flips, _, done, _ = run_episode(agent, graph, errors[i], epsilon=0.02, train=False)
        correction[i] = flips
        completed[i] = done
    return correction, completed


def run() -> list[dict]:
    agent = DQNAgent()
    agent.load(str(RL_MODEL_PATH))

    rows = []
    for d in DISTANCES:
        code = RepetitionCode(distance=d)
        for p in P_VALUES:
            rng = np.random.default_rng(cell_seed(d, p))
            errors = sample_errors(code, p, N_SHOTS, rng)

            rl_corr, rl_completed = rl_decode(agent, d, errors)
            final_values = (errors + rl_corr) % 2
            decoded_wrong = majority_vote_logical_readout(final_values) != 0
            # An incomplete episode (ran out of qubits without clearing
            # the syndrome) is exactly as much a decoding failure as a
            # wrong one; both count against the logical error rate.
            rl_ler = float(np.mean(decoded_wrong | ~rl_completed))

            row = {
                "distance": d, "p": p, "n_shots": N_SHOTS,
                "rl_ler": rl_ler,
                "rl_completion_rate": float(rl_completed.mean()),
                "rl_is_extrapolation": d not in TRAIN_DISTANCES_RL,
            }
            rows.append(row)
            print(f"d={d:2d} p={p:.2f}  rl={rl_ler:.4f}{'*' if row['rl_is_extrapolation'] else ' '}  "
                  f"completion={row['rl_completion_rate']:.2f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT_PATH}")
    return rows


if __name__ == "__main__":
    run()
