"""Training loop for the sequential RL decoder.

Uses two fixes for stable DQN convergence:

  1. Potential-based reward shaping (Ng, Harada & Russell 1999) using
     `DecodingEnv.potential()` (negative residual syndrome weight):
     dense per-step feedback instead of a reward that's silent until the
     syndrome fully clears.
  2. Double DQN (select the next action with the online network, evaluate
     it with the target network) to avoid the Q-value overestimation that
     plain DQN is prone to.

Also uses best-checkpoint deployment: DQN training is not guaranteed
monotonic, so the agent that's actually saved is whichever validation
checkpoint scored best during training, not necessarily the final one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .dqn_agent import DQNAgent, Transition
from .repetition_code import RepetitionCode, compute_syndrome, logical_error_rate, sample_errors
from .rl_env import DecodingEnv
from .rl_features import batch_features
from .tanner_graph import build_tanner_graph

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RL_MODEL_PATH = PROCESSED_DIR / "dqn_decoder.pt"
RL_TRAIN_LOG_PATH = PROCESSED_DIR / "rl_training_log.json"


def run_episode(agent: DQNAgent, graph, errors: np.ndarray, epsilon: float, train: bool,
                 max_steps: int | None = None) -> tuple[np.ndarray, float, bool, bool]:
    """Returns (final_correction, total_reward, done, logical_error)."""
    syndrome = compute_syndrome(errors[None, :])[0]
    env = DecodingEnv(graph, syndrome, true_error=errors)
    # Since candidate_actions() excludes already-flipped qubits (see
    # DecodingEnv.candidate_actions), an episode can run for at most
    # n_data steps before it either finishes or runs out of qubits to try.
    max_steps = max_steps or graph.n_data

    total_reward, steps, logical_error = 0.0, 0, False
    while not env.done() and steps < max_steps:
        candidates = env.candidate_actions()
        feats = batch_features(env, candidates)
        action_idx = agent.act(feats, epsilon=epsilon if train else 0.0)
        chosen_qubit = candidates[action_idx]
        chosen_features = feats[action_idx]

        potential_before = env.potential() if train else 0.0
        result = env.step(chosen_qubit)
        total_reward += result.reward
        if result.done:
            logical_error = result.info.get("logical_error", False)

        if train:
            potential_after = 0.0 if result.done else env.potential()
            shaped_reward = result.reward + agent.gamma * potential_after - potential_before
            next_candidates = env.candidate_actions() if not env.done() else []
            next_feats = batch_features(env, next_candidates)
            agent.remember(Transition(
                features=chosen_features, reward=shaped_reward,
                next_action_features=next_feats, done=env.done(),
            ))
        steps += 1

    return env.flipped, total_reward, env.done(), logical_error


def make_random_training_instance(rng: np.random.Generator):
    distance = int(rng.choice([5, 7, 9, 11]))
    p = float(rng.uniform(0.05, 0.3))
    code = RepetitionCode(distance=distance)
    errors = sample_errors(code, p, 1, rng)[0]
    graph = build_tanner_graph(distance)
    return graph, errors


def _make_fixed_validation_set(seed: int = 999, n_per_distance: int = 60):
    rng = np.random.default_rng(seed)
    pairs = []
    for d in (5, 7, 9, 11):
        code = RepetitionCode(distance=d)
        graph = build_tanner_graph(d)
        errors = sample_errors(code, 0.12, n_per_distance, rng)
        for e in errors:
            pairs.append((graph, e))
    return pairs


def evaluate_greedy(agent: DQNAgent, pairs, eval_epsilon: float = 0.02) -> dict:
    n_flips, completed, logical_errors = [], [], []
    for graph, errors in pairs:
        flips, _, done, logical_error = run_episode(agent, graph, errors, epsilon=eval_epsilon, train=False)
        n_flips.append(int(flips.sum()))
        completed.append(done)
        logical_errors.append(logical_error if done else True)
    return {
        "mean_flips": float(np.mean(n_flips)),
        "completion_rate": float(np.mean(completed)),
        "logical_error_rate": float(np.mean(logical_errors)),
    }


def train(n_episodes: int = 4000, seed: int = 0, eval_every: int = 200,
          target_update_every: int = 15) -> DQNAgent:
    rng = np.random.default_rng(seed)
    agent = DQNAgent(lr=1e-3, gamma=0.95)
    validation_pairs = _make_fixed_validation_set()

    epsilon_start, epsilon_end, epsilon_decay_episodes = 1.0, 0.05, int(n_episodes * 0.7)
    history = []
    t0 = time.time()
    best_score = (-1.0, 1.0, float("-inf"))  # (completion_rate, -logical_error_rate, -mean_flips)
    best_state = None

    for ep in range(n_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay_episodes)
        graph, errors = make_random_training_instance(rng)
        run_episode(agent, graph, errors, epsilon, train=True)

        for _ in range(4):
            agent.train_step(batch_size=64)
        if ep % target_update_every == 0:
            agent.update_target()

        if ep % eval_every == 0:
            metrics = evaluate_greedy(agent, validation_pairs)
            elapsed = time.time() - t0
            score = (metrics["completion_rate"], -metrics["logical_error_rate"], -metrics["mean_flips"])
            is_best = score > best_score
            if is_best:
                best_score = score
                best_state = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
            print(f"ep {ep:5d}/{n_episodes}  eps={epsilon:.2f}  "
                  f"val_completion={metrics['completion_rate']:.2f}  "
                  f"val_LER={metrics['logical_error_rate']:.3f}  "
                  f"val_mean_flips={metrics['mean_flips']:.2f}  "
                  f"{'* new best *' if is_best else ''}  elapsed={elapsed:.1f}s")
            history.append({"episode": ep, "epsilon": epsilon, **metrics, "is_best": is_best,
                             "elapsed_s": elapsed})

    final_metrics = evaluate_greedy(agent, validation_pairs)
    final_score = (final_metrics["completion_rate"], -final_metrics["logical_error_rate"],
                   -final_metrics["mean_flips"])
    if final_score > best_score:
        best_score = final_score
        best_state = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
    history.append({"episode": n_episodes, **final_metrics, "is_best": final_score >= best_score,
                     "elapsed_s": time.time() - t0})
    print(f"final validation: {final_metrics}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        agent.q_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)
    agent.save(str(RL_MODEL_PATH))
    RL_TRAIN_LOG_PATH.write_text(json.dumps(history, indent=2))
    print(f"saved best-validation checkpoint to {RL_MODEL_PATH}")
    return agent


if __name__ == "__main__":
    train()
