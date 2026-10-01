# JEV-style source review for GMT decision formulation

This review is a source audit of the pinned shallow clones.  No reference
backbone, Qwen model, or reference training code is copied into GMT.  The
references are used only to define the dynamic-option decision questions and
the calibration/uncertainty checks in this feasibility study.

| reference | pinned HEAD | license observed |
|---|---|---|
| [OpenJev](https://github.com/ejhshen/OpenJev) | `e279c44829370cbdc5c48784a912f38ffd73df60` | MIT |
| [OmniJev](https://github.com/tinnel123666888/OmniJev) | `14dbec4f71e194852c8d7b88ab36ef639493f400` | Apache-2.0 (repository notes the Qwen3-VL backbone has its own license) |
| [OneJev](https://github.com/OmniJev/OneJev) | `81ce62f1597c91e46767d4d02d6ac2e18534fe94` | Apache-2.0 |
| [reflex](https://github.com/kshetrajna12/reflex) | `231f896d818a62b94fec305ed553df1088486dcb` | MIT |
| [imajev](https://github.com/mohit67890/imajev) | `ccf586d43d2a580319b6535c893668904d909eb9` | Apache-2.0 |

## OpenJev (primary reference)

The audited implementation is in `src/openjev/models/decision/` and
`src/openjev/training/supervised.py`.

* `DecisionHead` has separate `question_projection` and
  `option_projection` layers.  A question is the state/query and each option
  is an independently encoded candidate.
* `OptionSetInteractor` applies masked, pre-norm multi-head self-attention to
  the complete option set.  The option mask is carried through every block;
  padded options are zeroed and cannot receive probability.
* The terminal compatibility is bilinear: a transformed question and a
  transformed option are reduced by a scaled elementwise product.  The head
  returns logits, log probabilities, probabilities, and the mask.
* `decision_cross_entropy` is full-distribution CE reduced once per decision,
  with safe handling of masked `-inf` entries.  The target is a distribution,
  so the formulation supports an explicit NEW/abstain option as another
  valid option.

The correspondence used here is deliberately small: the current detection
and read-only GMT history form the **question/state**, while the active
Global IDs form the **dynamic options**.  The option set can change with every
frame and has no fixed class vocabulary.  Candidate interaction is therefore
tested with a permutation-equivariant set block rather than by assigning a
semantic meaning to an option position.

## OmniJev

The audited `mso/head.py`, `mso/branch.py`, `mso/infer.py`, and `mso/records.py`
separate the question from options and make choice, score, and abstention
outputs explicit.  The records and proper-score paths motivate reporting
probability quality in addition to top-1 accuracy.  Their multimodal/Qwen
backbone is outside this project and is not imported into GMT.

## OneJev, Reflex, and ImaJev

* OneJev's `qev/calibrate.py` and `train/common.py` provide the calibration
  pattern used here: fit a scalar temperature on a held-out calibration
  split, then report CE/NLL, Brier, and ECE.
* `reflex/src/reflex/calibration_head.py` conditions uncertainty calibration
  on option count, entropy, top-1/top-2 margin, and state size.  These are
  diagnostics for the verification head, not state actions.
* ImaJev's `vision_decision/backend.py` and `calibration.py` make UNKNOWN an
  explicit candidate and handle dynamic candidate sets and candidate-order
  robustness.  This motivates a separate NEW column and the distractor and
  permutation tests.

These observations define tests and interfaces only.  The GMT experiments
use its existing detector, ReID features, association Transformer, and
checkpoint; they do not alter pretrained weights or run on VisionTrack test
annotations.
