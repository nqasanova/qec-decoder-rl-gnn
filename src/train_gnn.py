"""Supervised training for the GNN decoder.

Training data is generated per code distance with `repetition_code.py`'s
exact classical simulator (fast, exact; see that module's docstring for
why that's not a shortcut taken for convenience here). The label for each
data qubit is simply whether it was actually flipped, standard practice
for ML-based QEC decoders (e.g. Torlai & Melko 2017's neural decoder is
trained the same way): predict the physical error directly from the
syndrome via supervised learning against simulated ground truth, since the
simulator gives you that ground truth for free.

Training mixes multiple code distances in the same run (not one model per
distance) specifically to test whether the GNN's weight-sharing actually
buys the generalization it's supposed to: the held-out distances in
`evaluate.py` are never trained on at all, only interpolated/extrapolated
to by a model trained on smaller and/or different distances.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from .gnn_decoder import GNNDecoder, PROCESSED_DIR, GNN_MODEL_PATH, build_and_compile
from .repetition_code import RepetitionCode, compute_syndrome, exact_map_decoder, logical_error_rate, sample_errors
from .tanner_graph import build_tanner_graph, node_features

TRAIN_LOG_PATH = PROCESSED_DIR / "gnn_training_log.json"


def make_batch(distance: int, p: float, n_shots: int, rng: np.random.Generator):
    code = RepetitionCode(distance=distance)
    errors = sample_errors(code, p, n_shots, rng)
    syndrome = compute_syndrome(errors)
    graph = build_tanner_graph(distance)
    feats = node_features(graph, syndrome)
    return graph, feats, errors, syndrome


def evaluate_gnn(model: GNNDecoder, distances: list[int], p: float, n_shots: int, rng: np.random.Generator) -> dict:
    from .gnn_decoder import predict_correction
    per_distance = {}
    for d in distances:
        graph, feats, errors, syndrome = make_batch(d, p, n_shots, rng)
        corr = predict_correction(model, graph, syndrome)
        ler = logical_error_rate(errors, corr)
        # Compare against the exact optimal decoder on the same shots, so
        # "how close to optimal is the GNN" is a same-data comparison.
        exact_corr = exact_map_decoder(syndrome)
        exact_ler = logical_error_rate(errors, exact_corr)
        per_distance[d] = {"gnn_ler": ler, "exact_ler": exact_ler}
    return per_distance


def train(train_distances=(3, 5, 7), eval_distances=(3, 5, 7, 9, 11),
          p_train_range=(0.03, 0.25), p_eval: float = 0.1,
          n_epochs: int = 30, batches_per_epoch: int = 40, shots_per_batch: int = 256,
          eval_shots: int = 4000, seed: int = 0) -> GNNDecoder:
    rng = np.random.default_rng(seed)
    model = build_and_compile(hidden=32, n_layers=3, lr=2e-3)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_losses = []
        for _ in range(batches_per_epoch):
            d = int(rng.choice(train_distances))
            p = float(rng.uniform(*p_train_range))
            graph, feats, errors, syndrome = make_batch(d, p, shots_per_batch, rng)

            targets = errors.astype(np.float32)  # (n_shots, n_data)
            feats_t = tf.constant(feats)
            norm_adj_t = tf.constant(graph.norm_adjacency)
            targets_t = tf.constant(targets)

            with tf.GradientTape() as tape:
                logits = model((feats_t, norm_adj_t), training=True)
                data_logits = tf.gather(logits, graph.data_node_ids, axis=1)
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_t, logits=data_logits))
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss.numpy()))

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            metrics = evaluate_gnn(model, list(eval_distances), p_eval, eval_shots, rng)
            elapsed = time.time() - t0
            summary = "  ".join(f"d={d}:LER={m['gnn_ler']:.4f}(exact={m['exact_ler']:.4f})"
                                 for d, m in metrics.items())
            print(f"epoch {epoch:3d}/{n_epochs}  loss={np.mean(epoch_losses):.4f}  "
                  f"elapsed={elapsed:.1f}s\n    {summary}")
            history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)),
                             "metrics": metrics, "elapsed_s": elapsed})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(GNN_MODEL_PATH))
    TRAIN_LOG_PATH.write_text(json.dumps(history, indent=2))
    print(f"saved GNN weights to {GNN_MODEL_PATH}")
    return model


def load_trained_model(hidden: int = 32, n_layers: int = 3, sample_distance: int = 5) -> GNNDecoder:
    model = build_and_compile(hidden=hidden, n_layers=n_layers)
    # Keras needs the model called once (to build variable shapes) before weights can load.
    graph, feats, _, syndrome = make_batch(sample_distance, 0.1, 4, np.random.default_rng(0))
    model((tf.constant(feats), tf.constant(graph.norm_adjacency)), training=False)
    model.load_weights(str(GNN_MODEL_PATH))
    return model


if __name__ == "__main__":
    train()
