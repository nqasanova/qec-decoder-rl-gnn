import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rl_env import DecodingEnv, make_env_from_errors
from src.rl_features import batch_features
from src.tanner_graph import build_tanner_graph


def test_no_error_starts_done():
    graph = build_tanner_graph(5)
    syndrome = np.zeros(graph.n_checks, dtype=np.uint8)
    env = DecodingEnv(graph, syndrome)
    assert env.done()
    assert env.potential() == 0.0


def test_single_flip_syndrome_needs_one_step_minimum():
    # Error on qubit 2 of a distance-5 code triggers checks 1 and 2 (0-indexed).
    errors = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(5, errors)
    assert not env.done()
    assert env.residual_syndrome.tolist() == [0, 1, 1, 0]
    # Flipping the actual error qubit clears both checks at once.
    result = env.step(2)
    assert env.done()
    assert result.done
    assert result.info["logical_error"] is False


def test_flipping_wrong_qubit_pushes_defect_instead_of_clearing():
    errors = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(5, errors)
    # Flipping qubit 1 (adjacent to check 1, which is active) doesn't clear
    # the syndrome; it should toggle check 0 on and check 1 off, "pushing"
    # the defect from check 1 to check 0, net residual weight unchanged.
    weight_before = env.residual_syndrome.sum()
    env.step(1)
    weight_after = env.residual_syndrome.sum()
    assert weight_after == weight_before  # defect moved, didn't vanish
    assert not env.done()


def test_double_flip_undoes_itself():
    # Flipping the same qubit twice is a physical no-op (X*X = I), so it
    # must exactly restore the residual syndrome to what it was before.
    errors = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(5, errors)
    original_residual = env.residual_syndrome.copy()
    env.step(1)
    assert not np.array_equal(env.residual_syndrome, original_residual)
    env.step(1)
    assert np.array_equal(env.residual_syndrome, original_residual)


def test_terminal_reward_reflects_logical_correctness():
    # Error on qubit 0 only (weight 1, well within distance-5 code's
    # correction radius): the true minimum-weight correction (flip qubit 0
    # itself) must succeed.
    errors = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(5, errors)
    result = env.step(0)  # flip the actual error qubit -> exact correction
    assert result.done
    assert result.info["logical_error"] is False
    assert result.reward > 0  # -1 (flip cost) + terminal_bonus should net positive


def test_above_threshold_error_can_genuinely_fool_minimum_weight_decoding():
    # A distance-3 code only guarantees correcting weight-1 errors
    # (floor((3-1)/2) = 1). A weight-2 error is *expected* to sometimes
    # fool even the optimal (minimum-weight) decoder; that's not a bug,
    # it's the whole reason logical error rate isn't zero and the reason
    # larger code distances matter. Here, true error [1,1,0] (weight 2)
    # produces syndrome [0,1]; the minimum-weight correction consistent
    # with that syndrome is [0,0,1] (weight 1), lower weight than the
    # truth, so it's what any minimum-weight decoder (RL, GNN, or exact)
    # picks, and it's genuinely the wrong answer here: it flips the
    # logical bit.
    errors = np.array([1, 1, 0], dtype=np.uint8)
    graph = build_tanner_graph(3)
    from src.repetition_code import compute_syndrome
    syndrome = compute_syndrome(errors[None, :])[0]
    assert syndrome.tolist() == [0, 1]
    env = DecodingEnv(graph, syndrome, true_error=errors)
    result = env.step(2)  # the minimum-weight consistent correction
    assert result.done
    assert result.info["logical_error"] is True


def test_candidate_actions_excludes_already_flipped_qubits():
    errors = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(5, errors)
    assert set(env.candidate_actions()) == {0, 1, 2, 3, 4}
    env.step(1)
    assert 1 not in env.candidate_actions()
    assert set(env.candidate_actions()) == {0, 2, 3, 4}


def test_episode_always_terminates_within_n_data_steps_even_with_worst_policy():
    # A policy that always picks the *last* remaining candidate (a
    # deliberately bad, non-greedy policy) must still terminate within
    # n_data steps now that repeats are excluded; this is what actually
    # rules out the infinite flip/unflip 2-cycle an earlier, unmasked
    # version of this environment could fall into.
    rng = np.random.default_rng(0)
    for d in (5, 7, 9, 11):
        errors = (rng.random(d) < 0.3).astype(np.uint8)
        env = make_env_from_errors(d, errors)
        steps = 0
        while not env.done() and steps < d:
            candidates = env.candidate_actions()
            env.step(candidates[-1])  # worst-case adversarial choice
            steps += 1
        assert steps <= d


def test_batch_features_shape_and_finiteness():
    errors = np.array([0, 1, 0, 1, 0, 0, 0], dtype=np.uint8)
    env = make_env_from_errors(7, errors)
    candidates = env.candidate_actions()
    feats = batch_features(env, candidates)
    assert feats.shape == (7, 6)
    assert np.isfinite(feats).all()
