import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pennylane_sim import run_pennylane_shots
from src.repetition_code import RepetitionCode, compute_syndrome, sample_errors


def test_pennylane_circuit_matches_classical_simulator_random_errors():
    """The core claim `repetition_code.py` relies on: for this code, a real
    gate-level noisy circuit (X errors + CNOT-based parity extraction,
    simulated in PennyLane) produces exactly the same syndromes as the
    classical XOR shortcut. If this test ever failed, the classical
    shortcut used everywhere else in this project would be unjustified.
    """
    rng = np.random.default_rng(7)
    code = RepetitionCode(distance=5)
    errors = sample_errors(code, 0.25, 40, rng)
    classical_synd = compute_syndrome(errors)
    quantum_synd = run_pennylane_shots(5, errors)
    assert np.array_equal(classical_synd, quantum_synd)


def test_pennylane_circuit_matches_on_every_single_flip():
    """Exhaustive check over every single-qubit error pattern for a small
    code: the 5 basis error patterns that matter most for a distance-5
    code (it's designed to correct up to 2 errors, so single flips are the
    most common nontrivial case)."""
    d = 5
    errors = np.eye(d, dtype=np.uint8)  # one flip per row, on qubit i
    classical_synd = compute_syndrome(errors)
    quantum_synd = run_pennylane_shots(d, errors)
    assert np.array_equal(classical_synd, quantum_synd)


def test_pennylane_circuit_matches_on_no_error():
    d = 5
    errors = np.zeros((1, d), dtype=np.uint8)
    classical_synd = compute_syndrome(errors)
    quantum_synd = run_pennylane_shots(d, errors)
    assert np.array_equal(classical_synd, quantum_synd)
    assert not quantum_synd.any()
