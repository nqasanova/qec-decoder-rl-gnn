"""Tests for the TensorFlow GNN decoder.

Kept in tests/gnn/, physically separate from tests/rl/ (the PyTorch DQN
tests): this project's environment segfaults if TensorFlow and PyTorch
are both imported into the same process (see src/evaluate.py's
docstring for the full story), so these two test suites must be run as
separate `pytest` invocations, never together. See the README's "Run the
tests" section for the exact commands.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gnn_decoder import GNNDecoder, build_and_compile, predict_correction
from src.tanner_graph import build_tanner_graph, node_features


def test_forward_pass_shape():
    graph = build_tanner_graph(7)
    syndrome = np.zeros((4, graph.n_checks), dtype=np.uint8)
    feats = node_features(graph, syndrome)
    model = GNNDecoder(hidden=16, n_layers=2)
    logits = model((tf.constant(feats), tf.constant(graph.norm_adjacency)), training=False)
    assert logits.shape == (4, graph.n_nodes)


def test_predict_correction_shape_and_binary():
    graph = build_tanner_graph(5)
    rng = np.random.default_rng(0)
    syndrome = (rng.random((10, graph.n_checks)) < 0.5).astype(np.uint8)
    model = build_and_compile(hidden=8, n_layers=2)
    corr = predict_correction(model, graph, syndrome)
    assert corr.shape == (10, graph.n_data)
    assert set(np.unique(corr)).issubset({0, 1})


def test_model_generalizes_to_a_different_graph_size_without_retraining():
    # The whole point of the GCN architecture (shared weights, no
    # fixed-size input layer) is that the same model object can be called
    # on graphs of different sizes. This just checks that mechanically
    # works, not that predictions are good on the untrained size (that's
    # what data/reports/decoder_eval.md's extrapolation columns cover).
    model = build_and_compile(hidden=8, n_layers=2)
    for d in (3, 7, 11):
        graph = build_tanner_graph(d)
        syndrome = np.zeros((2, graph.n_checks), dtype=np.uint8)
        corr = predict_correction(model, graph, syndrome)
        assert corr.shape == (2, d)


def test_training_step_reduces_loss_on_a_fixed_batch():
    graph = build_tanner_graph(5)
    rng = np.random.default_rng(1)
    n = 64
    errors = (rng.random((n, graph.n_data)) < 0.15).astype(np.uint8)
    from src.repetition_code import compute_syndrome
    syndrome = compute_syndrome(errors)
    feats = node_features(graph, syndrome)

    model = build_and_compile(hidden=16, n_layers=2, lr=1e-2)
    feats_t = tf.constant(feats)
    norm_adj_t = tf.constant(graph.norm_adjacency)
    targets_t = tf.constant(errors.astype(np.float32))

    def step():
        with tf.GradientTape() as tape:
            logits = model((feats_t, norm_adj_t), training=True)
            data_logits = tf.gather(logits, graph.data_node_ids, axis=1)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_t, logits=data_logits))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return float(loss.numpy())

    losses = [step() for _ in range(30)]
    assert losses[-1] < losses[0]
