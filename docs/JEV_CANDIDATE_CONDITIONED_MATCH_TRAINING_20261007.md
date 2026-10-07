# Candidate-conditioned MATCH screening — 2026-10-07

This is the first corrected-source candidate experiment. It is an offline
screening result only: no runtime closed loop, paper selection, or Full-H8
authorization follows from it.

## Provenance and split

- Source commit: `a390682724f79d1482e7caf1008300b6e5634fec`
- Seed: `20261003`
- Horizon: `8`
- Training: video07 / `00020court3`, 1,703 MATCH records
- Validation: video06 / `00017court1`, 2,600 MATCH records
- Optimizer settings: AdamW, batch 128, learning rate `1e-3`, 20 epochs
- GMT checkpoint SHA: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`

The train/validation traces were freshly generated from the same corrected
source and have action-mismatch count zero. Candidate IDs are metadata only;
they are not features or fixed classifier classes. Candidate targets come from
independent isolated frozen-evidence rollouts, not from native action labels.

## Validation result

| Method | NLL | Best-candidate accuracy | Brier | ECE | Expected utility | Greedy utility | Δ greedy vs native |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native action binding | — | — | — | — | — | 1.251789 | — |
| Candidate MLP | 1.272657 | 0.997308 | 0.062597 | 0.345773 | 0.474930 | 1.272366 | +0.020577 |
| Candidate JEV | 1.270739 | 0.993462 | 0.060617 | 0.343108 | 0.465050 | 1.271404 | +0.019615 |
| Feasible-set oracle | — | — | — | — | — | 1.277174 | +0.025385 |
| Uniform random | — | — | — | — | — | -0.759866 | -2.011655 |

The greedy utility is the relevant number for a runtime argmax policy. Both
models show a small positive offline signal over the native binding, but the
headroom to the feasible-set oracle is only about `0.005` utility. The result
is therefore encouraging as a pipeline check, not evidence of a robust final
improvement.

## Scope limits and next gate

The earlier candidate audit found zero current-GT recall for native
REACTIVATION candidates, so this experiment covers MATCH only. The next gate
is a legal runtime candidate binding on held-out video01: use the scorer only
over the current GMT candidate set, verify runtime state-feature parity and
candidate selection parity, then compare GMT OFF / candidate MLP / candidate
JEV on the same sequence. Full H8 remains unauthorized until that closed loop
passes.

Exact dataset/checkpoint hashes and all reported values are in
`reports/JEV_RNG_V4/CANDIDATE_CONDITIONED_MATCH_TRAINING_20261007.json`.
