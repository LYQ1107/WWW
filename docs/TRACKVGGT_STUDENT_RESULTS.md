# Conditional student status

Status: NOT_EXECUTED_RELATION_GATE_NOT_PASSED_YET. Remaining DINO
temporal controls are still required before the final gate disposition.

No TrackVGGT student has been trained or deployed. There is no student
checkpoint, trained-seed result, learned full-video correction, or student FPS.
The existing TinyRelationStudent/IdentityRelationAdapter classes and their
zero-initialization, residual, masking and proposal aggregation contracts are
implementation prerequisites; passing those contracts is not student efficacy.

The prespecified GT-only, KD-only, GT+KD, matched DINO-KD and GMT-teacher student
comparisons remain conditional on valid independent teacher relations. If a
teacher passes that gate, all required comparisons, three actual trained seeds,
paired inference RNG, unchanged full DEV5 inputs and complete system evaluation
must be executed before reporting a student outcome. Original native/Prior
inference seeds are not student-training seeds. The fixed temperature is2.0;
online students accept native GMT evidence only, with zero-initialized residual
bounded by1.0 and no VGGT/DINO/cache dependency.

Completed evidence does not yet justify this expenditure: primary V3 AG-C is
below ROI/native, QD is below the matched nativeMLP, local crops are below
matched DINO, and the VGGT cycle comparison has no certified new ranking fix.
No low-quality teacher distribution is manufactured into a successful
distillation experiment. This status will be revisited after the remaining
independently registered teacher experiments finish; it does not cancel them.
