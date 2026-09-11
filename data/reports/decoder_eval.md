# Decoder evaluation report

Evaluated all 4 decoders on the same 3000 freshly sampled shots per (distance, physical error rate) cell, 5 distances x 6 error rates = 30 cells.

GNN trained on distances (3, 5, 7); RL trained on distances (5, 7, 9, 11). A `*` marks a cell where that decoder is extrapolating to a code distance it never trained on.

## Logical error rate by distance and physical error rate

| d | p | exact (optimal) | GNN | RL | naive |
|---|---|---|---|---|---|
| 3 | 0.03 | 0.0020 | 0.0020 | 0.0403* | 0.0240 |
| 3 | 0.06 | 0.0140 | 0.0140 | 0.0580* | 0.0563 |
| 3 | 0.10 | 0.0290 | 0.0290 | 0.1073* | 0.1117 |
| 3 | 0.15 | 0.0603 | 0.0603 | 0.1450* | 0.1540 |
| 3 | 0.20 | 0.0937 | 0.0937 | 0.1923* | 0.1907 |
| 3 | 0.25 | 0.1570 | 0.1570 | 0.2453* | 0.2640 |
| 5 | 0.03 | 0.0003 | 0.0003 | 0.0017 | 0.0037 |
| 5 | 0.06 | 0.0013 | 0.0013 | 0.0093 | 0.0110 |
| 5 | 0.10 | 0.0063 | 0.0063 | 0.0297 | 0.0287 |
| 5 | 0.15 | 0.0267 | 0.0267 | 0.0647 | 0.0597 |
| 5 | 0.20 | 0.0630 | 0.0630 | 0.1110 | 0.1030 |
| 5 | 0.25 | 0.1040 | 0.1040 | 0.1670 | 0.1530 |
| 7 | 0.03 | 0.0000 | 0.0000 | 0.0017 | 0.0000 |
| 7 | 0.06 | 0.0003 | 0.0010 | 0.0040 | 0.0027 |
| 7 | 0.10 | 0.0020 | 0.0047 | 0.0180 | 0.0070 |
| 7 | 0.15 | 0.0090 | 0.0120 | 0.0400 | 0.0193 |
| 7 | 0.20 | 0.0363 | 0.0453 | 0.0827 | 0.0540 |
| 7 | 0.25 | 0.0673 | 0.0810 | 0.1390 | 0.1040 |
| 9 | 0.03 | 0.0000 | 0.0000* | 0.0067 | 0.0000 |
| 9 | 0.06 | 0.0000 | 0.0000* | 0.0167 | 0.0003 |
| 9 | 0.10 | 0.0013 | 0.0010* | 0.0427 | 0.0033 |
| 9 | 0.15 | 0.0057 | 0.0077* | 0.1033 | 0.0093 |
| 9 | 0.20 | 0.0157 | 0.0210* | 0.1773 | 0.0293 |
| 9 | 0.25 | 0.0500 | 0.0540* | 0.2620 | 0.0663 |
| 11 | 0.03 | 0.0000 | 0.0000* | 0.0033 | 0.0000 |
| 11 | 0.06 | 0.0000 | 0.0000* | 0.0153 | 0.0000 |
| 11 | 0.10 | 0.0003 | 0.0003* | 0.0537 | 0.0010 |
| 11 | 0.15 | 0.0013 | 0.0043* | 0.1067 | 0.0033 |
| 11 | 0.20 | 0.0133 | 0.0180* | 0.1863 | 0.0193 |
| 11 | 0.25 | 0.0270 | 0.0350* | 0.2657 | 0.0400 |

## Summary: mean (decoder LER / optimal LER), restricted to the 17/30 cells where exact LER >= 0.005 (below that, ratios of two near-zero numbers are not a meaningful comparison). Lower is better, 1.0 = optimal.

| Decoder | On trained distances | On unseen (extrapolated) distances |
|---|---|---|
| GNN | 1.07 | 1.28 |
| RL | 6.49 | 2.77 |
| naive (local, no learning) | 2.17 | n/a |

## Summary: mean absolute logical error rate across all 30 cells (the plain, unfiltered comparison)

| Decoder | Mean LER |
|---|---|
| exact (optimal) | 0.0262 |
| GNN | 0.0281 |
| RL | 0.0899 |
| naive (local, no learning) | 0.0506 |
