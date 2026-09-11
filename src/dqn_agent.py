"""A small DQN agent that scores candidate qubit-flip actions.

An early version of this agent, trained on the raw -1-per-flip reward with
plain DQN, was unstable (see `train_rl.py`'s docstring). Potential-based
reward shaping and Double DQN fixed it: a generic property of
sparse-terminal-reward DQN on this kind of sequential graph-clearing task,
not something specific to one environment.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_features import N_FEATURES


class QNetwork(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Transition:
    features: np.ndarray
    reward: float
    next_action_features: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 30_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.95, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer()

    def act(self, candidate_features: np.ndarray, epsilon: float) -> int:
        n = candidate_features.shape[0]
        if n == 0:
            raise ValueError("no candidate actions available")
        if random.random() < epsilon:
            return random.randrange(n)
        with torch.no_grad():
            x = torch.as_tensor(candidate_features, dtype=torch.float32, device=self.device)
            q = self.q_net(x)
        return int(torch.argmax(q).item())

    def remember(self, t: Transition) -> None:
        self.buffer.push(t)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)

        feats = torch.as_tensor(np.stack([t.features for t in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(feats)

        with torch.no_grad():
            targets = rewards.clone()
            for i, t in enumerate(batch):
                if not t.done and len(t.next_action_features) > 0:
                    next_feats = torch.as_tensor(t.next_action_features, dtype=torch.float32, device=self.device)
                    best_idx = torch.argmax(self.q_net(next_feats))  # Double DQN: select w/ online net
                    next_q = self.target_net(next_feats)[best_idx]   # evaluate w/ target net
                    targets[i] += self.gamma * next_q

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
