# JEV reference architecture audit

This audit used shallow source clones only.  Exact repository HEADs,
licenses, and inspected paths are in `reports/JEV_REFERENCE_PROVENANCE.json`.
No JEV checkpoint or large model was downloaded, and no reference source is
copied into GMT.

## Observed design principles

**OmniJev.** `mso/head.py` represents each option separately, uses a
question-conditioned FiLM-style gate, emits option logits, and keeps a
temperature per primitive type. Choice is a probability distribution with an
explicit abstain mass; Noul is binary; Score is a cumulative-link ordinal
head with ordered thresholds. Log score, Brier, RPS and binary log loss are
implemented as proper objectives. `branch.py` shares the state prefix and
isolates option branches, so option ordering does not leak into the state.
`v04.py` adds backbone-derived option confidence features.

**OpenJev.** `OptionSetInteractor` applies masked, permutation-equivariant
self-attention to the complete runtime option set. A question projection and
option projection are combined before interaction, then bilinear compatibility
(`q · e_j`) produces a masked softmax over the options. `decision_cross_entropy`
trains the complete distribution, including soft targets; the RL module adds
REINFORCE-A and reference-KL terms. The set mask and shared prefix are a
direct fit for a current detection as question and active Global IDs as
dynamic options.

**OneJev.** Calibration is post-hoc and keyed by primitive and option count;
the training configuration combines cross entropy and Brier. `train/shared.py`
encodes one state prefix and branches multiple questions without autoregressive
text generation. Calibration is separated from training data.

**Reflex.** The calibration head explicitly accepts option count, entropy,
top-two margin, state size and question type. These are useful uncertainty
features for GMT association, but they do not replace candidate-set modeling.

**ImaJev.** The vision decision backend directly reads candidate logits,
supports an explicit `UNKNOWN` option, rotates candidate order and averages
the semantic candidate scores, and calibrates temperature by decision type and
option count. Its readout is a structured candidate head rather than a text
generation path.

## Synthesis

| Design property | OmniJev | OpenJev | OneJev | Reflex | ImaJev | Adopt for JEV-GMT? |
|---|---|---|---|---|---|---|
| dynamic candidates | yes | yes | yes | yes | yes | **yes** |
| candidate-set interaction | gated option scorer | masked set attention | shared question branches | calibration features | direct candidate readout | **yes, set attention** |
| permutation equivariance | branch isolation | explicit | order augmentation | feature-level | cyclic order test | **yes** |
| explicit UNKNOWN/NEW | abstain | mask/option set | option schema | no fixed option | UNKNOWN | **yes, NEW** |
| Choice head | yes | bilinear softmax | primitive choice | no | direct logits | **yes** |
| binary verification | Noul | supervised binary possible | Noul | calibration | decision type | **yes** |
| ordinal risk | cumulative-link Score | not central | score primitive | no | ordinal support | **yes** |
| probability calibration | type temperature | logits/proper loss | primitive/count temperature | dedicated head | type/count temperature | **yes** |
| proper scoring rule | log/Brier/RPS | CE/Brier/KL | CE+Brier | calibration losses | NLL/calibration | **yes** |
| shared state encoding | yes | yes | yes | state features | shared prefix | **yes** |
| autoregressive generation | optional reference backbone | no decision generation | no | no | no | **no** |
| model size | backbone-dependent head | configurable head | backbone-dependent | small head | adapter/readout | **<2M head** |

The recommended JEV-GMT component is therefore a small structured decision
network: a 128-dimensional state encoder, a 128-dimensional candidate
encoder, two permutation-equivariant set-attention layers, a masked bilinear
Choice head, a binary Verification head and a cumulative-link Risk head. It
must operate on GMT representations and numeric tracker evidence, never raw
RGB, Qwen, a VLM, an LLM, or autoregressive text. The audit will only build
this component after the predeclared causal gate is a **strong GO**.
