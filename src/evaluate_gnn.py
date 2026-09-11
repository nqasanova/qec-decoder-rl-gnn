"""Evaluate the exact decoder, the GNN decoder, and the naive baseline
(everything that does NOT need PyTorch) across the full (distance,
p) grid, and write results to data/processed/eval_gnn_results.json.

Run as a **separate process** from the RL evaluation (`evaluate_rl.py`)
deliberately: importing both TensorFlow and PyTorch in the same Python
process segfaults in this project's environment (a real, reproducible
crash hit while building this project, not a hypothetical concern).
Rather than fight a low-level C++ runtime conflict between two bundled
OpenMP/MKL builds, the clean fix is to never let both frameworks load in
one process; `evaluate.py` runs this script and `evaluate_rl.py` as two
subprocesses and merges their JSON output. Each cell uses a seed derived
only from (distance, p), not from iteration order or a shared RNG
stream, specifically so this script and `evaluate_rl.py` sample the
exact same error shots for the same cell despite never sharing memory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from .eval_config import DISTANCES, N_SHOTS, P_VALUES, PROCESSED_DIR, cell_seed
from .gnn_decoder import predict_correction
from .naive_baseline import naive_local_decoder
from .repetition_code import RepetitionCode, compute_syndrome, exact_map_decoder, logical_error_rate, sample_errors
from .tanner_graph import build_tanner_graph
from .train_gnn import load_trained_model

OUT_PATH = PROCESSED_DIR / "eval_gnn_results.json"
TRAIN_DISTANCES_GNN = (3, 5, 7)


def run() -> list[dict]:
    gnn_model = load_trained_model()
    rows = []
    for d in DISTANCES:
        code = RepetitionCode(distance=d)
        graph = build_tanner_graph(d)
        for p in P_VALUES:
            rng = np.random.default_rng(cell_seed(d, p))
            errors = sample_errors(code, p, N_SHOTS, rng)
            syndrome = compute_syndrome(errors)

            exact_corr = exact_map_decoder(syndrome)
            exact_ler = logical_error_rate(errors, exact_corr)

            gnn_corr = predict_correction(gnn_model, graph, syndrome)
            gnn_ler = logical_error_rate(errors, gnn_corr)

            naive_corr = naive_local_decoder(syndrome)
            naive_ler = logical_error_rate(errors, naive_corr)

            row = {
                "distance": d, "p": p, "n_shots": N_SHOTS,
                "exact_ler": exact_ler, "gnn_ler": gnn_ler,
                "gnn_is_extrapolation": d not in TRAIN_DISTANCES_GNN,
                "naive_ler": naive_ler,
            }
            rows.append(row)
            print(f"d={d:2d} p={p:.2f}  exact={exact_ler:.4f}  "
                  f"gnn={gnn_ler:.4f}{'*' if row['gnn_is_extrapolation'] else ' '}  naive={naive_ler:.4f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT_PATH}")
    return rows


if __name__ == "__main__":
    run()
