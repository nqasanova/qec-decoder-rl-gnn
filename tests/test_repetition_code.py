import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.repetition_code import (
    RepetitionCode, compute_syndrome, exact_map_decoder,
    logical_error_rate, majority_vote_logical_readout, residual_syndrome_rate,
    sample_errors,
)


def test_syndrome_of_no_error_is_all_zero():
    errors = np.zeros((5, 7), dtype=np.uint8)
    synd = compute_syndrome(errors)
    assert not synd.any()


def test_syndrome_single_flip_triggers_adjacent_checks():
    errors = np.zeros((1, 7), dtype=np.uint8)
    errors[0, 3] = 1
    synd = compute_syndrome(errors)
    # qubit 3 flipped -> checks 2 (btwn q2,q3) and 3 (btwn q3,q4) trigger, others don't.
    expected = np.zeros((1, 6), dtype=np.uint8)
    expected[0, 2] = 1
    expected[0, 3] = 1
    assert np.array_equal(synd, expected)


def test_exact_decoder_fully_clears_syndrome_always():
    rng = np.random.default_rng(1)
    code = RepetitionCode(distance=9)
    errors = sample_errors(code, 0.3, 2000, rng)
    synd = compute_syndrome(errors)
    corr = exact_map_decoder(synd)
    assert residual_syndrome_rate(errors, corr) == 0.0


def test_exact_decoder_matches_brute_force_optimum_all_patterns():
    d = 7
    all_errors = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.uint8)
    synd = compute_syndrome(all_errors)
    corr = exact_map_decoder(synd)

    # Precompute the brute-force minimum weight per syndrome.
    best_weight = {}
    for cand in all_errors:
        s = tuple(np.bitwise_xor(cand[:-1], cand[1:]).tolist())
        w = int(cand.sum())
        if s not in best_weight or w < best_weight[s]:
            best_weight[s] = w

    for s_row, c_row in zip(synd, corr):
        s = tuple(s_row.tolist())
        assert int(c_row.sum()) == best_weight[s]


def test_exact_decoder_achieves_zero_logical_error_below_half_weight():
    # A correction that exactly reverses the true error always succeeds.
    rng = np.random.default_rng(2)
    code = RepetitionCode(distance=5)
    errors = sample_errors(code, 0.1, 5000, rng)
    synd = compute_syndrome(errors)
    corr = exact_map_decoder(synd)
    # Not every shot succeeds (that's the whole point of a nonzero logical
    # error rate), but it should be small at low p and, since d=5 corrects
    # up to floor((d-1)/2)=2 errors, any shot with <=2 flipped qubits must
    # be corrected exactly right (this is the actual distance guarantee).
    low_weight = errors.sum(axis=1) <= 2
    ler_low_weight = logical_error_rate(errors[low_weight], corr[low_weight])
    assert ler_low_weight == 0.0


def test_majority_vote_readout():
    vals = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 0, 1]], dtype=np.uint8)
    out = majority_vote_logical_readout(vals)
    assert list(out) == [0, 1]


def test_logical_error_rate_increases_with_p():
    rng = np.random.default_rng(3)
    code = RepetitionCode(distance=7)
    rates = []
    for p in (0.02, 0.1, 0.25):
        errors = sample_errors(code, p, 20000, rng)
        synd = compute_syndrome(errors)
        corr = exact_map_decoder(synd)
        rates.append(logical_error_rate(errors, corr))
    assert rates[0] < rates[1] < rates[2]


def test_larger_distance_suppresses_logical_error_at_fixed_p():
    # This is the actual "resource efficiency" tradeoff: more physical
    # qubits per logical qubit (larger distance) should give a lower
    # logical error rate at the same physical error rate, for p well
    # below the code's threshold.
    rng = np.random.default_rng(4)
    p = 0.08
    rates = []
    for d in (3, 5, 7, 9):
        code = RepetitionCode(distance=d)
        errors = sample_errors(code, p, 30000, rng)
        synd = compute_syndrome(errors)
        corr = exact_map_decoder(synd)
        rates.append(logical_error_rate(errors, corr))
    assert rates == sorted(rates, reverse=True), f"expected monotonic decrease, got {rates}"
