"""Gate-level circuit simulation of the repetition code's syndrome
extraction, built with PennyLane: data qubits, ancilla qubits, explicit
bit-flip errors as X gates, and CNOT-based parity extraction, exactly the
circuit a real device would run.

This exists to validate `repetition_code.py`'s classical XOR-based
simulator against an actual quantum circuit, not to replace it: for this
specific code (X errors only, Z-type stabilizers only), the two are
mathematically guaranteed to agree, since everything involved is diagonal
in the computational basis and the "quantum" circuit never leaves it. The
value of building it anyway is exactly what `test_pennylane_sim.py`
checks: that the equivalence used to justify the classical shortcut in
`repetition_code.py` actually holds, rather than just asserting it, and
having a real circuit-simulation entry point is also what makes this
project's approach extend to codes (like the surface code) where the
shortcut genuinely stops applying, since X and Z errors interact there and
you can no longer avoid quantum circuit simulation.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def build_syndrome_circuit(distance: int):
    """Returns a PennyLane QNode that: prepares |0>^d on the data qubits,
    applies an X gate to each data qubit whose corresponding `flip` input
    is 1, runs the standard CNOT-based parity-check circuit into d-1
    ancilla qubits, and returns computational-basis samples of every wire
    (data qubits first, then ancillas). Because every step is diagonal in
    the computational basis (X gates and CNOTs targeting Z-basis parity),
    the circuit never produces superposition, deliberately, since we're
    validating the classical shortcut, not testing coherent QEC.
    """
    n_data = distance
    n_checks = distance - 1
    n_wires = n_data + n_checks
    data_wires = list(range(n_data))
    ancilla_wires = list(range(n_data, n_wires))

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, shots=1)
    def circuit(flips):
        for i in data_wires:
            qml.RX(np.pi * flips[i], wires=i)  # flips[i] in {0,1}: RX(pi)=X, RX(0)=I
        for check_idx, (d0, d1) in enumerate(zip(data_wires[:-1], data_wires[1:])):
            a = ancilla_wires[check_idx]
            qml.CNOT(wires=[d0, a])
            qml.CNOT(wires=[d1, a])
        return [qml.sample(qml.PauliZ(w)) for w in range(n_wires)]

    return circuit, data_wires, ancilla_wires


def run_pennylane_shots(distance: int, errors: np.ndarray) -> np.ndarray:
    """Run the actual gate-level circuit once per shot in `errors`
    (n_shots, distance) and return the measured syndrome (n_shots, distance-1),
    read off the ancilla qubits exactly as a real device would.

    This is deliberately the slow, literal path (one circuit execution per
    shot, no batching); it exists as a small-scale cross-check, not as
    the bulk data-generation path (that's `repetition_code.sample_errors` +
    `compute_syndrome`, which is exact and orders of magnitude faster for
    the thousands of shots GNN/RL training needs).
    """
    circuit, data_wires, ancilla_wires = build_syndrome_circuit(distance)
    n_shots = errors.shape[0]
    syndromes = np.zeros((n_shots, distance - 1), dtype=np.uint8)
    for i in range(n_shots):
        samples = circuit(errors[i].astype(float))
        # qml.sample(PauliZ) returns a length-1 array (shots=1) of +1/-1;
        # map to bit: +1 -> 0, -1 -> 1.
        bits = np.array([(1 - int(np.asarray(s).reshape(-1)[0])) // 2 for s in samples])
        ancilla_bits = bits[len(data_wires):]
        syndromes[i] = ancilla_bits
    return syndromes
