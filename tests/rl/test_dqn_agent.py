"""Tests for the PyTorch DQN decoding agent.

Kept in tests/rl/, physically separate from tests/gnn/ (the TensorFlow GNN
tests); see that file's docstring for why: this project's environment
segfaults if TensorFlow and PyTorch are both imported into the same
process, so these two suites must be run as separate `pytest` invocations.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dqn_agent import DQNAgent, Transition
from src.rl_features import N_FEATURES


def test_act_returns_valid_index():
    agent = DQNAgent()
    feats = np.random.randn(5, N_FEATURES).astype(np.float32)
    assert 0 <= agent.act(feats, epsilon=0.0) < 5
    assert 0 <= agent.act(feats, epsilon=1.0) < 5


def test_act_raises_on_no_candidates():
    agent = DQNAgent()
    feats = np.zeros((0, N_FEATURES), dtype=np.float32)
    try:
        agent.act(feats, epsilon=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_train_step_requires_minimum_batch():
    agent = DQNAgent()
    assert agent.train_step(batch_size=64) is None


def test_train_step_reduces_loss_toward_fixed_target():
    agent = DQNAgent(lr=1e-2)
    rng = np.random.default_rng(0)
    fixed_features = np.ones(N_FEATURES, dtype=np.float32)
    for _ in range(200):
        agent.remember(Transition(
            features=fixed_features + rng.normal(0, 0.01, N_FEATURES).astype(np.float32),
            reward=-1.0,
            next_action_features=np.zeros((0, N_FEATURES), dtype=np.float32),
            done=True,
        ))
    losses = [loss for _ in range(50) if (loss := agent.train_step(batch_size=32)) is not None]
    assert losses[-1] < losses[0]


def test_bootstrapped_target_actually_uses_the_target_network():
    # A focused regression check for the Double DQN fix (train_rl.py's
    # docstring explains why it mattered): the TD target for a non-
    # terminal transition must depend on target_net's parameters, not
    # just q_net's. If a future refactor accidentally dropped
    # `self.target_net(...)` in favor of reusing `self.q_net(...)` for
    # both selection and evaluation (silently reverting to vanilla DQN),
    # this test would stop failing to detect it: perturbing only
    # target_net must change the loss for a fixed non-terminal transition.
    agent = DQNAgent(gamma=0.9)
    next_feats = np.random.randn(4, N_FEATURES).astype(np.float32)
    fixed_transition = Transition(
        features=np.zeros(N_FEATURES, dtype=np.float32), reward=0.0,
        next_action_features=next_feats, done=False,
    )
    for _ in range(40):
        agent.remember(fixed_transition)

    loss_before = agent.train_step(batch_size=32)

    # Reset and perturb ONLY target_net (q_net untouched) with a fresh
    # agent so the comparison isn't confounded by the first train_step's
    # gradient update.
    agent2 = DQNAgent(gamma=0.9)
    agent2.q_net.load_state_dict(agent.q_net.state_dict())
    agent2.target_net.load_state_dict(agent.target_net.state_dict())
    with torch.no_grad():
        for p in agent2.target_net.parameters():
            p.add_(1.0)
    for _ in range(40):
        agent2.remember(fixed_transition)
    loss_after_perturbing_target = agent2.train_step(batch_size=32)

    assert loss_before is not None and loss_after_perturbing_target is not None
    assert loss_before != loss_after_perturbing_target


def test_save_and_load_roundtrip(tmp_path):
    agent = DQNAgent()
    feats = np.random.randn(3, N_FEATURES).astype(np.float32)
    before = agent.q_net(torch.as_tensor(feats)).detach().numpy()

    path = tmp_path / "model.pt"
    agent.save(str(path))

    agent2 = DQNAgent()
    agent2.load(str(path))
    after = agent2.q_net(torch.as_tensor(feats)).detach().numpy()

    assert np.allclose(before, after)
