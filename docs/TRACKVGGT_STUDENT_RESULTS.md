# Conditional student status

Status: CONDITIONAL_NOT_EXECUTED_RELATION_GATE_FAILED. All56 registered
teacher runs and both independently paired temporal controls are complete.
The actual gate is NO_UNIQUE_VGGT_TEACHER_EVIDENCE; see
results/trackvggt_teacher_gate_v1.json.

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

Completed evidence does not justify this expenditure: primary V3 AG-C is
below ROI/native, QD is below the matched nativeMLP, local crops are below
matched DINO, and the VGGT cycle comparison has no certified new ranking fix.
No low-quality teacher distribution is manufactured into a successful
distillation experiment. DINO temporal0/.1 also finishes without a certified
new ranking repair (R1=.21/.20). The conditional student branch is explicitly
unexecuted under the locked design, not counted as a trained negative student,
engineering deployment success or three-seed result. All fresh full native system tests have now completed independently; their
baseline metrics do not become trained-student results or satisfy a student
success claim.
