"""Unified command-line entry point.

    python -m src.cli train-gnn --epochs 80        # train the GNN decoder (TensorFlow)
    python -m src.cli train-rl --episodes 4000      # train the RL decoder (PyTorch)
    python -m src.cli evaluate                      # compare all 4 decoders (runs as subprocesses)
    python -m src.cli plot                          # plot both training curves + the LER-vs-p comparison
    python -m src.cli all                            # steps above, in order

Every subcommand imports its own dependencies lazily, inside the function
body, rather than at module level. This matters here specifically: this
project's environment segfaults if TensorFlow and PyTorch are ever both
imported into the same process (see `evaluate.py`'s docstring), so `cli.py`
itself must never trigger both imports in one invocation; lazy imports
keep each subcommand's process footprint to exactly the one framework it
needs.
"""
from __future__ import annotations

import argparse


def cmd_train_gnn(args):
    from .train_gnn import train
    train(n_epochs=args.epochs, seed=args.seed)


def cmd_train_rl(args):
    from .train_rl import train
    train(n_episodes=args.episodes, seed=args.seed)


def cmd_evaluate(args):
    from .evaluate import main
    main()


def cmd_plot(args):
    from .plot_results import plot_all
    plot_all()


def cmd_all(args):
    cmd_train_gnn(args)
    cmd_train_rl(args)
    cmd_evaluate(args)
    cmd_plot(args)


def main():
    parser = argparse.ArgumentParser(description="QEC decoder RL/GNN pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-gnn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train-rl")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)

    sub.add_parser("evaluate")
    sub.add_parser("plot")

    p = sub.add_parser("all")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    {
        "train-gnn": cmd_train_gnn,
        "train-rl": cmd_train_rl,
        "evaluate": cmd_evaluate,
        "plot": cmd_plot,
        "all": cmd_all,
    }[args.command](args)


if __name__ == "__main__":
    main()
