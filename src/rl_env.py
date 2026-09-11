"""A sequential-decision RL environment for repetition-code decoding.

Instead of predicting the whole correction in one shot (the GNN's
approach), the RL agent decodes by repeatedly choosing one data qubit to
flip, observing how the *residual* syndrome changes, until every check is
satisfied. This is an "insert one graph-local action at a time, watch a
residual quantity shrink" structure: a sequential decision over a graph,
driven by a residual constraint being cleared at minimum cost.

Reward: -1 per flip (each physical correction has a real cost, more gates,
more chances to introduce a *new* error, so fewer flips is more
resource-efficient), shaped with the residual syndrome weight as a
potential function, plus a one-off terminal
bonus/penalty for whether the *decoded logical bit* actually comes out
correct, which the per-step shaping reward can't see, since a
syndrome-clearing correction can still be the *wrong* one (see
`repetition_code.py`'s docstring on why only two error classes are
consistent with any given syndrome).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tanner_graph import TannerGraph, build_tanner_graph


@dataclass
class StepResult:
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class DecodingEnv:
    def __init__(self, graph: TannerGraph, syndrome: np.ndarray, true_error: np.ndarray | None = None,
                 logical_bit: int = 0, terminal_bonus: float = 5.0):
        assert syndrome.shape == (graph.n_checks,)
        self.graph = graph
        self.original_syndrome = syndrome.copy()
        self.residual_syndrome = syndrome.copy().astype(np.uint8)
        self.flipped = np.zeros(graph.n_data, dtype=np.uint8)
        self.true_error = true_error  # only used for reward/eval, never exposed as a feature
        self.logical_bit = logical_bit
        self.terminal_bonus = terminal_bonus
        self.n_flips = 0

    def done(self) -> bool:
        return not self.residual_syndrome.any()

    def potential(self) -> float:
        """Negative residual syndrome weight, shrinks toward 0 as the
        agent clears checks, giving dense per-step feedback instead of a
        reward that's silent until the very last check clears."""
        return -float(self.residual_syndrome.sum())

    def candidate_actions(self) -> list[int]:
        """Every data qubit *not already flipped this episode*. Excluding
        already-flipped qubits isn't just a convenience mask: the only two
        flip patterns that can ever fully clear a given syndrome are a
        fixed pair of complementary bitstrings (see
        `repetition_code.exact_map_decoder`'s docstring), and every qubit
        belongs to exactly one of the two, so no valid minimum-weight
        correction ever needs to flip the same qubit twice. Allowing
        immediate re-flips only opens the door to an under-trained greedy
        policy settling into a flip/unflip 2-cycle and never terminating
        (a real failure this project's training run hit; see
        `train_rl.py`'s module docstring); excluding them instead
        guarantees every episode terminates within at most `n_data` steps,
        whether or not the agent ends up choosing the right subset.
        """
        return [q for q in range(self.graph.n_data) if not self.flipped[q]]

    def step(self, qubit_idx: int) -> StepResult:
        assert not self.done(), "episode already finished"
        for check_node in self.graph.data_neighbor_checks[qubit_idx]:
            check_idx = check_node - self.graph.n_data
            self.residual_syndrome[check_idx] ^= 1
        self.flipped[qubit_idx] ^= 1
        self.n_flips += 1

        reward = -1.0
        done = self.done()
        info = {}
        if done and self.true_error is not None:
            from .repetition_code import logical_error_rate
            ler = logical_error_rate(self.true_error[None, :], self.flipped[None, :], self.logical_bit)
            info["logical_error"] = bool(ler > 0)
            reward += -self.terminal_bonus if info["logical_error"] else self.terminal_bonus
        return StepResult(reward=reward, done=done, info=info)


def make_env_from_errors(distance: int, errors: np.ndarray, logical_bit: int = 0) -> DecodingEnv:
    from .repetition_code import compute_syndrome
    graph = build_tanner_graph(distance)
    syndrome = compute_syndrome(errors[None, :])[0]
    return DecodingEnv(graph, syndrome, true_error=errors, logical_bit=logical_bit)
