"""The distance-d bit-flip repetition code: the simplest nontrivial quantum
error-correcting code, and the standard first testbed for QEC decoders
before moving to the surface code (which is literally a repetition code
glued together in two directions to also correct phase-flip errors).

Encoding: a logical qubit |b>_L is encoded as |b b b ... b> across d
physical data qubits. Independent bit-flip (X) errors occur on each data
qubit with probability p. Syndrome extraction measures the d-1 stabilizers
Z_i Z_{i+1} (does data qubit i agree with its neighbour?) without
collapsing the encoded logical information. Decoding = given the syndrome,
guess which qubits were flipped and apply a correcting X to each.

Design choice: this module simulates errors and syndromes with plain
classical XOR arithmetic, not a quantum circuit. That's not a shortcut
taken for convenience: it's mathematically exact for this specific code.
Bit-flip (X) errors and Z-basis parity-check measurements are both
diagonal in the computational basis, so the whole process commutes and
reduces exactly to classical bit arithmetic; `src/pennylane_sim.py`
verifies this equivalence against an actual gate-level noisy quantum
circuit. This shortcut stops being available the moment a code needs to
correct *both* X and Z errors with non-commuting stabilizers (e.g. the
surface code); see the README's "What's next" for why PennyLane's role
here is a deliberately-scoped foundation, not the whole story.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepetitionCode:
    distance: int  # number of physical data qubits (must be odd, >= 3)

    def __post_init__(self):
        assert self.distance >= 3 and self.distance % 2 == 1, \
            "repetition code distance must be an odd integer >= 3"

    @property
    def n_data(self) -> int:
        return self.distance

    @property
    def n_checks(self) -> int:
        return self.distance - 1


def sample_errors(code: RepetitionCode, p: float, n_shots: int, rng: np.random.Generator) -> np.ndarray:
    """Independent bit-flip errors on each data qubit. Returns a
    (n_shots, d) uint8 array, 1 = this qubit was flipped."""
    return (rng.random((n_shots, code.n_data)) < p).astype(np.uint8)


def compute_syndrome(errors: np.ndarray) -> np.ndarray:
    """syndrome[i] = error[i] XOR error[i+1] for i in 0..d-2, i.e. did
    adjacent data qubits disagree. Shape (n_shots, d-1)."""
    return np.bitwise_xor(errors[:, :-1], errors[:, 1:])


def exact_map_decoder(syndrome: np.ndarray) -> np.ndarray:
    """The optimal (maximum-likelihood, for any error rate p < 0.5)
    decoder for the repetition code, computed exactly in O(d) per shot.

    Any correction consistent with a given syndrome is fully determined by
    one free bit (whether qubit 0 is corrected) via the recurrence
    correction[i+1] = correction[i] XOR syndrome[i]; the two candidates
    (starting from 0 or from 1) are complements of each other. Minimum
    weight picks the maximum-likelihood one whenever p < 0.5 (a standard
    QEC result: lower-weight error patterns are always more likely under
    i.i.d. bit-flip noise), which is exactly what real repetition/surface
    code decoders (e.g. minimum-weight perfect matching) compute; on this
    1-D code there's no need for a general graph-matching library, since
    the optimal decoder has this closed form.
    """
    n_shots, n_checks = syndrome.shape
    d = n_checks + 1

    candidate0 = np.zeros((n_shots, d), dtype=np.uint8)
    candidate0[:, 1:] = np.bitwise_xor.accumulate(syndrome, axis=1)
    candidate1 = 1 - candidate0

    weight0 = candidate0.sum(axis=1)
    weight1 = candidate1.sum(axis=1)
    use1 = weight1 < weight0

    correction = np.where(use1[:, None], candidate1, candidate0)
    return correction.astype(np.uint8)


def majority_vote_logical_readout(final_qubit_values: np.ndarray) -> np.ndarray:
    """Decode the logical bit from the final (post-error, post-correction)
    physical qubit values by majority vote, exactly how a repetition
    code's logical value is read out in practice. Shape (n_shots, d) -> (n_shots,)."""
    return (final_qubit_values.mean(axis=1) > 0.5).astype(np.uint8)


def logical_error_rate(errors: np.ndarray, correction: np.ndarray, logical_bit: int = 0) -> float:
    """The headline metric for any decoder: after applying `correction` to
    qubits that suffered `errors` (both start from an encoded `logical_bit`),
    what fraction of shots does majority-vote readout disagree with the
    original logical bit? This is well-defined regardless of whether
    `correction` happens to exactly satisfy the syndrome, which matters
    because a learned decoder (GNN/RL) isn't guaranteed to always find a
    syndrome-consistent correction the way the exact decoder is.
    """
    final_values = (logical_bit + errors + correction) % 2
    decoded = majority_vote_logical_readout(final_values)
    return float(np.mean(decoded != logical_bit))


def residual_syndrome_rate(errors: np.ndarray, correction: np.ndarray) -> float:
    """Fraction of shots where the correction doesn't even fully cancel the
    syndrome (a stricter, decoder-quality diagnostic separate from the
    logical error rate, which only cares about the final logical bit)."""
    residual = (errors + correction) % 2
    residual_syndrome = compute_syndrome(residual)
    return float(np.mean(residual_syndrome.any(axis=1)))
