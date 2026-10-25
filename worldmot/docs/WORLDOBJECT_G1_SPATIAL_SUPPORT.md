# World Object G1 — TEST spatial support

G1 compares the two camera observations in the Ω pointmap after explicit
original-RGB-to-model-coordinate box mapping. It does not call the rectangle a
segmentation mask: depth-cluster support carries an ambiguity code and can
abstain.

Across TEST (22 scenes, 29,019 timestamps):

- matched detector observations: 510,963;
- same-person cross-view pairs: 170,868;
- same-person support-consistent pairs: 143,972 (84.2592%);
- different-person pairs: 5,386,541;
- different-person ambiguous pairs: 279,074 (5.1810%);
- unmatched observations: 37,620.

The full per-observation audit is `G1_object_support_all22.csv`; the compact
summary is `G1_object_support_summary.json`. GT is used only to label this
diagnostic after inference, not to construct the support.

