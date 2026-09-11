"""Plots for this project's two training runs and the final decoder
comparison. Kept in its own module (only needs matplotlib + json, no
TensorFlow or PyTorch) so it can run after both training/eval subprocess
scripts have already written their JSON output, without needing either
framework loaded itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"


def plot_gnn_training(out_path: Path | None = None) -> Path:
    history = json.loads((PROCESSED_DIR / "gnn_training_log.json").read_text())
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs, loss, marker="o", color="#4C72B0")
    ax1.set_ylabel("training loss")
    ax1.set_title("GNN decoder training")
    ax1.grid(alpha=0.3)

    colors = {"3": "#55A868", "5": "#C44E52", "7": "#8172B2", "9": "#CCB974", "11": "#64B5CD"}
    for d_str, color in colors.items():
        d = int(d_str)
        vals = [h["metrics"][d_str]["gnn_ler"] for h in history if d_str in h.get("metrics", {})]
        if vals:
            ax2.plot(epochs[:len(vals)], vals, marker="o", color=color, label=f"d={d}")
    ax2.set_ylabel("validation logical error rate")
    ax2.set_xlabel("training epoch")
    ax2.set_yscale("log")
    ax2.legend(ncol=5, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_path or (REPORTS_DIR / "gnn_training_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def plot_rl_training(out_path: Path | None = None) -> Path:
    history = json.loads((PROCESSED_DIR / "rl_training_log.json").read_text())
    episodes = [h["episode"] for h in history if "completion_rate" in h]
    ler = [h["logical_error_rate"] for h in history if "completion_rate" in h]
    completion = [h["completion_rate"] for h in history if "completion_rate" in h]
    best_ep = [h["episode"] for h in history if h.get("is_best")]
    best_ler = [h["logical_error_rate"] for h in history if h.get("is_best")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(episodes, ler, marker="o", color="#4C72B0", label="validation LER")
    ax1.scatter(best_ep, best_ler, color="#C44E52", zorder=5, marker="*", s=120, label="new best checkpoint")
    ax1.set_ylabel("logical error rate")
    ax1.set_title("RL decoder training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(episodes, completion, marker="o", color="#55A868")
    ax2.set_ylabel("completion rate")
    ax2.set_xlabel("training episode")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_path or (REPORTS_DIR / "rl_training_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def plot_decoder_comparison(out_path: Path | None = None) -> Path:
    rows = json.loads((REPORTS_DIR / "decoder_eval.json").read_text())
    distances = sorted(set(r["distance"] for r in rows))

    fig, axes = plt.subplots(1, len(distances), figsize=(4 * len(distances), 4), sharey=True)
    if len(distances) == 1:
        axes = [axes]

    colors = {"exact_ler": "#333333", "gnn_ler": "#4C72B0", "rl_ler": "#C44E52", "naive_ler": "#999999"}
    labels = {"exact_ler": "exact (optimal)", "gnn_ler": "GNN", "rl_ler": "RL", "naive_ler": "naive"}
    styles = {"exact_ler": "-", "gnn_ler": "--", "rl_ler": "--", "naive_ler": ":"}

    for ax, d in zip(axes, distances):
        cell_rows = sorted([r for r in rows if r["distance"] == d], key=lambda r: r["p"])
        p_vals = [r["p"] for r in cell_rows]
        for key in ("exact_ler", "gnn_ler", "rl_ler", "naive_ler"):
            y = [max(r[key], 1e-4) for r in cell_rows]  # floor for log-scale plotting
            ax.plot(p_vals, y, styles[key], marker="o", markersize=4, color=colors[key], label=labels[key])
        ax.set_yscale("log")
        ax.set_title(f"distance {d}")
        ax.set_xlabel("physical error rate p")
        ax.grid(alpha=0.3, which="both")

    axes[0].set_ylabel("logical error rate (log scale)")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Decoder comparison: logical error rate vs. physical error rate")
    fig.tight_layout()

    out_path = out_path or (REPORTS_DIR / "decoder_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def plot_all():
    plot_gnn_training()
    plot_rl_training()
    plot_decoder_comparison()


if __name__ == "__main__":
    plot_all()
