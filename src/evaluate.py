"""Orchestrates the full decoder comparison: runs `evaluate_gnn.py` (exact
+ GNN + naive, needs TensorFlow) and `evaluate_rl.py` (RL, needs PyTorch)
as two **separate subprocesses**, then merges their per-cell results into
one report.

Why two subprocesses instead of one script that imports everything: this
project's environment reproducibly segfaults if TensorFlow and PyTorch
are both imported into the same process (confirmed while building this
project, not a theoretical worry: constructing a `torch.nn.Module`
after TensorFlow has initialized crashes with SIGSEGV here). Rather than
paper over that with a try/except that would just crash anyway, or drop
one of the two frameworks, this runs each model in its own clean process
and reconciles the results afterward. Both
subprocess scripts derive their per-cell error samples from the same
`cell_seed(distance, p)` function (see `evaluate_gnn.py`), so despite
never sharing memory, they evaluate on identical shots; the merge below
is a fair apples-to-apples comparison, not two different draws.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .evaluate_gnn import DISTANCES, OUT_PATH as GNN_OUT_PATH, P_VALUES
from .evaluate_rl import OUT_PATH as RL_OUT_PATH

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"


def run_subprocess(module: str) -> None:
    print(f"--- running python -m {module} (separate process) ---")
    result = subprocess.run([sys.executable, "-m", module], cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")


def merge(gnn_rows: list[dict], rl_rows: list[dict]) -> list[dict]:
    rl_by_key = {(r["distance"], r["p"]): r for r in rl_rows}
    merged = []
    for g in gnn_rows:
        key = (g["distance"], g["p"])
        r = rl_by_key[key]
        merged.append({**g, "rl_ler": r["rl_ler"], "rl_completion_rate": r["rl_completion_rate"],
                        "rl_is_extrapolation": r["rl_is_extrapolation"]})
    return merged


def write_report(rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "decoder_eval.json").write_text(json.dumps(rows, indent=2))

    lines = ["# Decoder evaluation report\n\n"]
    lines.append(f"Evaluated all 4 decoders on the same {rows[0]['n_shots']} freshly sampled shots "
                 f"per (distance, physical error rate) cell, {len(DISTANCES)} distances x "
                 f"{len(P_VALUES)} error rates = {len(rows)} cells.\n\n")
    lines.append("GNN trained on distances (3, 5, 7); RL trained on distances (5, 7, 9, 11). "
                 "A `*` marks a cell where that decoder is extrapolating to a code distance it "
                 "never trained on.\n\n")

    lines.append("## Logical error rate by distance and physical error rate\n\n")
    lines.append("| d | p | exact (optimal) | GNN | RL | naive |\n|---|---|---|---|---|---|\n")
    for r in rows:
        gnn_mark = "*" if r["gnn_is_extrapolation"] else ""
        rl_mark = "*" if r["rl_is_extrapolation"] else ""
        lines.append(f"| {r['distance']} | {r['p']:.2f} | {r['exact_ler']:.4f} | "
                     f"{r['gnn_ler']:.4f}{gnn_mark} | {r['rl_ler']:.4f}{rl_mark} | {r['naive_ler']:.4f} |\n")

    # Ratio-of-LERs is a misleading summary when exact_ler is near zero (a
    # handful of the low-p, small-distance cells above are 0.0000 vs
    # 0.0003, technically a huge "ratio" between two essentially-perfect
    # results). Restrict the ratio summary to cells with a non-negligible
    # exact LER (>= 0.005) where the comparison is actually meaningful, and
    # separately report plain mean absolute LER across every cell, which
    # isn't sensitive to this and is the primary headline number.
    RATIO_THRESHOLD = 0.005

    def avg_ratio(key: str, extrap_key: str, extrapolation: bool) -> float | None:
        vals = [r[key] / r["exact_ler"] for r in rows
                if r["exact_ler"] >= RATIO_THRESHOLD and r[extrap_key] == extrapolation]
        return float(np.mean(vals)) if vals else None

    def avg_abs_ler(key: str, extrap_key: str, extrapolation: bool | None = None) -> float:
        vals = [r[key] for r in rows if extrapolation is None or r[extrap_key] == extrapolation]
        return float(np.mean(vals))

    lines.append(f"\n## Summary: mean (decoder LER / optimal LER), restricted to the "
                 f"{sum(1 for r in rows if r['exact_ler'] >= RATIO_THRESHOLD)}/{len(rows)} cells "
                 f"where exact LER >= {RATIO_THRESHOLD} (below that, ratios of two near-zero "
                 f"numbers are not a meaningful comparison). Lower is better, 1.0 = optimal.\n\n")
    lines.append("| Decoder | On trained distances | On unseen (extrapolated) distances |\n"
                 "|---|---|---|\n")
    g_in, g_out = avg_ratio("gnn_ler", "gnn_is_extrapolation", False), avg_ratio("gnn_ler", "gnn_is_extrapolation", True)
    r_in, r_out = avg_ratio("rl_ler", "rl_is_extrapolation", False), avg_ratio("rl_ler", "rl_is_extrapolation", True)
    lines.append(f"| GNN | {g_in:.2f} | {g_out:.2f} |\n")
    lines.append(f"| RL | {r_in:.2f} | {r_out:.2f} |\n")
    naive_vals = [r["naive_ler"] / r["exact_ler"] for r in rows if r["exact_ler"] >= RATIO_THRESHOLD]
    lines.append(f"| naive (local, no learning) | {float(np.mean(naive_vals)):.2f} | n/a |\n")

    lines.append(f"\n## Summary: mean absolute logical error rate across all {len(rows)} cells "
                 f"(the plain, unfiltered comparison)\n\n")
    lines.append("| Decoder | Mean LER |\n|---|---|\n")
    lines.append(f"| exact (optimal) | {avg_abs_ler('exact_ler', None):.4f} |\n")
    lines.append(f"| GNN | {avg_abs_ler('gnn_ler', None):.4f} |\n")
    lines.append(f"| RL | {avg_abs_ler('rl_ler', None):.4f} |\n")
    lines.append(f"| naive (local, no learning) | {avg_abs_ler('naive_ler', None):.4f} |\n")

    (REPORTS_DIR / "decoder_eval.md").write_text("".join(lines))
    print(f"\nwrote report to {REPORTS_DIR / 'decoder_eval.md'}")


def main():
    run_subprocess("src.evaluate_gnn")
    run_subprocess("src.evaluate_rl")
    gnn_rows = json.loads(GNN_OUT_PATH.read_text())
    rl_rows = json.loads(RL_OUT_PATH.read_text())
    merged = merge(gnn_rows, rl_rows)
    write_report(merged)


if __name__ == "__main__":
    main()
