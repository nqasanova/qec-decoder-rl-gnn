"""The Tanner graph of the repetition code: a bipartite graph with one
node per data qubit and one node per parity-check (syndrome bit), edges
connecting each check to the two data qubits it involves. This is the
standard graph representation used by real QEC decoders (both classical
belief-propagation decoders and neural ones) because it makes the code's
actual constraint structure explicit, rather than throwing away structure
by flattening the syndrome into an unstructured feature vector, which is
exactly why it's the right input representation for a graph neural
network here, and why the same GNN weights generalize across code
distances (see `gnn_decoder.py`): the graph grows, but each node's local
neighbourhood looks the same regardless of the code's overall size.

    data_0 --- check_0 --- data_1 --- check_1 --- data_2 --- ... --- data_{d-1}

Node features:
  - data nodes: no informative feature at input time (the decoder isn't
    told which qubits were actually flipped; that's what it has to
    infer), so they get a constant placeholder feature.
  - check nodes: the observed syndrome bit (0 or 1).

This module builds the normalized adjacency matrix used by the GNN's
message-passing layers (`gnn_decoder.py`) and the neighbour lookup tables
used by the RL agent's per-qubit features (`rl_decoder.py`), so both
decoders read the graph the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TannerGraph:
    distance: int
    n_data: int
    n_checks: int
    n_nodes: int
    data_node_ids: np.ndarray      # indices of data nodes, 0..n_data-1
    check_node_ids: np.ndarray     # indices of check nodes, n_data..n_nodes-1
    adjacency: np.ndarray          # (n_nodes, n_nodes) symmetric 0/1 adjacency
    norm_adjacency: np.ndarray     # symmetric-normalized adjacency + self-loops, for GCN layers
    data_neighbor_checks: list     # for each data qubit i, list of check node ids touching it
    check_neighbor_data: list      # for each check node, the two data qubit ids it touches


def build_tanner_graph(distance: int) -> TannerGraph:
    n_data = distance
    n_checks = distance - 1
    n_nodes = n_data + n_checks

    data_ids = np.arange(n_data)
    check_ids = np.arange(n_data, n_nodes)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    data_neighbor_checks = [[] for _ in range(n_data)]
    check_neighbor_data = [[] for _ in range(n_checks)]

    for check_idx in range(n_checks):
        check_node = check_ids[check_idx]
        d0, d1 = check_idx, check_idx + 1
        adjacency[check_node, d0] = 1.0
        adjacency[d0, check_node] = 1.0
        adjacency[check_node, d1] = 1.0
        adjacency[d1, check_node] = 1.0
        data_neighbor_checks[d0].append(int(check_node))
        data_neighbor_checks[d1].append(int(check_node))
        check_neighbor_data[check_idx] = [d0, d1]

    # Symmetric GCN normalization (Kipf & Welling 2017): D^-1/2 (A+I) D^-1/2.
    a_hat = adjacency + np.eye(n_nodes, dtype=np.float32)
    deg = a_hat.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    norm_adjacency = (d_inv_sqrt[:, None] * a_hat) * d_inv_sqrt[None, :]

    return TannerGraph(
        distance=distance, n_data=n_data, n_checks=n_checks, n_nodes=n_nodes,
        data_node_ids=data_ids, check_node_ids=check_ids,
        adjacency=adjacency, norm_adjacency=norm_adjacency.astype(np.float32),
        data_neighbor_checks=data_neighbor_checks, check_neighbor_data=check_neighbor_data,
    )


def node_features(graph: TannerGraph, syndrome: np.ndarray) -> np.ndarray:
    """syndrome: (n_shots, n_checks) -> node features (n_shots, n_nodes, 2):
    feature 0 = "is this a check node", feature 1 = the check's syndrome
    bit (0 for data nodes). Two features is enough for a GCN to tell data
    and check nodes apart and to see the observed syndrome, without baking
    in anything about which qubits were actually flipped.
    """
    n_shots = syndrome.shape[0]
    feats = np.zeros((n_shots, graph.n_nodes, 2), dtype=np.float32)
    feats[:, graph.check_node_ids, 0] = 1.0
    feats[:, graph.check_node_ids, 1] = syndrome.astype(np.float32)
    return feats
