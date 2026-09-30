# Test/GT policy

The VisionTrack `test.json` file is read-only evidence. Its SHA256 is recorded
in `audit/manifests/baseline_provenance.json` and in every run manifest that
uses it. The audit does not edit annotations, remove objects, alter frame
order, or derive labels from predictions.

The three A0 sanity scenes are selected mechanically from the annotation-count
distribution, with ties resolved by scene name. The resulting subset JSON is a
new index containing the original records for those scenes; it is not a
replacement for the official test set and is never used for official scoring.

The A1/A4 observation table uses Hungarian matching solely for descriptive
diagnostics, accepting a pair only at IoU >= 0.5. It is not the MOTChallenge,
HOTA, IDF1, AssA, MOTA, CVMA, or CVIDF1 evaluator and cannot change official
metrics. No test GT is used to tune a threshold or select a released result.

All official baseline numbers are copied from the completed release-code
evaluation and remain immutable. Any audit-run prediction or metric is stored
with its exact command, environment switches, checkpoint link, and annotation
hash.
