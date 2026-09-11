"""A graph neural network decoder for the repetition code, trained by
supervised learning: given a syndrome, predict which physical data qubits
were flipped.

Implemented as a small stack of Graph Convolutional layers (Kipf &
Welling 2017: H' = sigma((D^-1/2 (A+I) D^-1/2) @ H @ W)) written directly
against TensorFlow's tensor ops, rather than pulling in a dedicated GNN
library (spektral, tf-gnn, ...). That keeps this project's dependency
footprint to exactly what's in requirements.txt and makes the
message-passing step fully transparent: the propagation rule is three
lines of matrix multiplication, not a black-box layer import.

Because the same weight matrices apply to a graph of any size (the model
only ever sees a per-node feature vector and an adjacency-derived
propagation matrix, never a fixed-size flattened input), one trained model
generalizes across code distances without retraining.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
GNN_MODEL_PATH = PROCESSED_DIR / "gnn_decoder.weights.h5"


class GraphConvLayer(keras.layers.Layer):
    """H' = activation(norm_adj @ H @ W + b). norm_adj is passed in at call
    time (not a layer weight) since it's a different, fixed matrix per
    graph: this project's graphs are the same topology per code distance
    but a different one is used per distance, and this layer needs to work
    for whichever one it's given.
    """

    def __init__(self, units: int, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.w = self.add_weight(shape=(feature_dim, self.units), initializer="glorot_uniform",
                                  trainable=True, name="w")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name="b")

    def call(self, h, norm_adj):
        # h: (batch, n_nodes, feature_dim), norm_adj: (n_nodes, n_nodes)
        propagated = tf.einsum("ij,bjf->bif", norm_adj, h)
        out = tf.einsum("bif,fo->bio", propagated, self.w) + self.b
        return self.activation(out)


class GNNDecoder(keras.Model):
    def __init__(self, hidden: int = 32, n_layers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.gc_layers = [GraphConvLayer(hidden, activation="relu") for _ in range(n_layers - 1)]
        self.gc_out = GraphConvLayer(1, activation=None)  # per-node logit

    def call(self, inputs):
        h, norm_adj = inputs
        for layer in self.gc_layers:
            h = layer(h, norm_adj)
        logits = self.gc_out(h, norm_adj)  # (batch, n_nodes, 1)
        return tf.squeeze(logits, axis=-1)  # (batch, n_nodes)


def predict_correction(model: GNNDecoder, graph, syndrome: np.ndarray) -> np.ndarray:
    """Run the model and slice out just the data-qubit logits (check-node
    outputs aren't used), thresholded at 0.5 probability to get a binary
    correction of shape (n_shots, n_data)."""
    from .tanner_graph import node_features

    feats = node_features(graph, syndrome)
    logits = model((feats, graph.norm_adjacency), training=False).numpy()
    data_logits = logits[:, graph.data_node_ids]
    probs = 1.0 / (1.0 + np.exp(-data_logits))
    return (probs > 0.5).astype(np.uint8)


def build_and_compile(hidden: int = 32, n_layers: int = 3, lr: float = 1e-3) -> GNNDecoder:
    model = GNNDecoder(hidden=hidden, n_layers=n_layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss=keras.losses.BinaryCrossentropy(from_logits=True))
    return model
