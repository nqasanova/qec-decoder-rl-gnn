# Dataset card: repetition-code error samples

This project doesn't use a downloaded dataset: every error pattern,
syndrome, and training/evaluation shot is generated deterministically by
code in this repo, rather than letting "dataset" imply something external.

## The code family (`src/repetition_code.py`, `src/tanner_graph.py`)

Five distances of the bit-flip repetition code: **d = 3, 5, 7, 9, 11**
(always odd, `n_data = d` physical qubits, `n_checks = d - 1` parity
checks). Its Tanner graph is a simple path, `data_0 - check_0 - data_1 -
check_1 - ... - data_{d-1}`, which keeps the ground-truth decoder
closed-form (see `repetition_code.exact_map_decoder`'s docstring) and lets
`tests/test_repetition_code.py` check it against a brute-force search over
every one of the 2^d error patterns for d = 7, not just spot-check it.

**Limitation:** a real device's noise isn't i.i.d. per-qubit bit-flip
noise, and the repetition code only protects against one error type (X).
It's the standard first testbed for QEC decoding research precisely
because it's simple enough to have an exact, cheap ground-truth decoder to
train and evaluate against, not because it's what a real backend would
run. The surface code (which corrects both X and Z errors by gluing two
repetition codes together) is the natural next step; see the README's
"What's next".

## Error samples (`repetition_code.sample_errors`)

Independent per-qubit bit-flip errors, each qubit flipped with probability
`p`, generated with `numpy.random.Generator` from an explicit seed, so
nothing here is hidden global RNG state. Two disjoint uses:

- **Training** (`train_gnn.py`, `train_rl.py`): freshly sampled every
  batch/episode from an RNG stream seeded once per run (`--seed`), never
  persisted to disk, and never reused for evaluation.
- **Evaluation** (`eval_config.py`'s `cell_seed(distance, p)`): a
  deterministic seed derived only from `(distance, p)`, so `evaluate_gnn.py`
  and `evaluate_rl.py`, which run as separate subprocesses and never
  share memory (see the README's "Design choices"), draw *exactly* the
  same 3,000 shots per cell for the same `(distance, p)`. This is what
  makes the head-to-head decoder comparison in
  [`data/reports/decoder_eval.md`](data/reports/decoder_eval.md) an
  apples-to-apples comparison rather than two different random draws that
  happen to be reported side by side.

Evaluation grid: 5 distances x 6 physical error rates (`p` in `{0.03,
0.06, 0.10, 0.15, 0.20, 0.25}`) x 3,000 shots = 90,000 total evaluated
shots, none of which overlap the training data (training draws from a
different, never-persisted RNG stream and, for the GNN, a different subset
of distances).

## Held-out generalization split

The GNN trains on distances **(3, 5, 7)** only; the RL agent trains on
distances **(5, 7, 9, 11)** only. Both are evaluated on **all five**
distances (3, 5, 7, 9, 11), so each decoder has at least one and up to
two distances in the evaluation grid it never trained on at all
(`*`-marked rows in `decoder_eval.md`). This is deliberate: the entire
point of a GCN's weight sharing is that it should generalize across graph
size without retraining, and the only reliable way to check that claim is
to actually hold distances out, not just report in-distribution numbers.

**Limitation:** "generalizes" here means "the same trained weights produce
a reasonable correction on a differently-sized Tanner graph," evaluated
only on distances 3-11. Nothing here says how far this keeps working (d =
101? d = 1001?); that would need training/eval infrastructure this
project doesn't build.

## PennyLane cross-check (`src/pennylane_sim.py`)

Not a dataset in the usual sense, but worth documenting here: a small,
independent gate-level circuit simulation (data qubits, ancillas, explicit
X-gate errors, CNOT-based parity extraction) used only to verify that
`repetition_code.py`'s fast classical XOR simulator agrees exactly with an
actual quantum circuit, on the same error patterns. `test_pennylane_sim.py`
checks this on every single-qubit-flip pattern plus random multi-qubit
patterns. It is not used to generate any of the training or evaluation
data above; that would be needlessly slow, since it's one PennyLane
circuit execution per shot, versus a single vectorized XOR for the whole
batch.
