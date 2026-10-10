# Phase XVI frozen four-slot reader trial

P0 qualified this trial using926 strict TRAIN identity/view/64-frame clusters,
including496 high-risk clusters. These counts identify raw evidence and a
cosine summary gap; they do not demonstrate correct native recovery.

The primary policy remains Phase XV V3 F_full LAST1500, seed20261009. The
detector, VFCE, native committed Gallery/hits/Bank, legal candidate rules,
four neural history slots, architecture and all weights remain frozen.
No candidate IDs are dropped or fabricated.

| Reader | Additional stored vectors per ID | Four-slot interpretation |
| --- | ---: | --- |
| A original | 0 | last, whole mean, own-camera mean, other-camera mean |
| B recent | ≤4 | last four actual committed observations |
| C temporal | ≤16 | latest segment, earliest anchor, two diverse remaining segments |
| D camera | ≤16 | own latest/earliest, other latest/earliest segments |
| E combination | ≤16 | last actual, earliest temporal, own latest segment, other latest segment |

Segments contain at most eight chronological observations. A scene-frame
gap greater than8 or cosine to the running segment below0.75 starts a new
segment. Eviction retains the earliest anchor and the latest remaining
segments. E stores four recent vectors, six temporal sums and three sums
per camera. Each segment stores one vector and past provenance keys.
The per-reader bounds are distinct from the unchanged, unbounded native
Gallery. Shadow comparisons keep several readers for measurement only.

The exact positive denominator is frozen before measurement: strict TRAIN
MATCH queries with a real legal ID having a past owner anchor, at least
three observed target points, and P0 raw target cosine≥0.75. Normal and
high-risk queries both remain included. The negative denominator includes
all strict TRAIN GT-known MATCH queries with actual legal candidates;
activation of any certified pure wrong ID at cosine≥0.75 is counted.
Mixed histories with known wrong immutable owners receive an additional
diagnostic and are not silently converted to WHO negatives.

A reader must improve positive support by at least2 percentage points,
increase certified-wrong activation by at most1 point, and newly support
at least8 distinct strict clusters. When several pass, the preregistered
selection order is E/D/C/B. TRAIN14 remains an isolated duplicate-GT
diagnostic. Reused DEV17/18/19 never sets the threshold or reader selection.

Every reader is evaluated on the same actual query. WHO head scores and
the full joint policy, including availability/trust/Q4/reliability, are
reported separately. The alternative assignments are shadow predictions;
they do not execute Bank recovery or a future native branch. Original
scores and committed IDs must remain exact against P0, and temporary
runtime_details are restored before the actual posterior write.

REACT retains its actual untrained first-three-slot maximum cosine,
0.75 terminal, native Hungarian and genuine stale candidate list. The
fourth slot is still excluded by that frozen fallback. The original
slot-kind embeddings stay frozen while B–E change token interpretation;
this deliberate input change can expose limitations of the frozen decoder.
Failure here cannot prove that raw history contains no identity information.

The initial measurement implementation caused many scalar CUDA transfers.
Six partial neutral trajectories and their original logs/source hashes are
preserved under P1_initial_engineering. The batch-export implementation
must match every complete overlapping record, including all five readers'
joint/WHO logits, assignments and certification diagnostics. These are
engineering executions, not extra independent samples or research versions.

Read/forward latency is CUDA-synchronized but comes from a cached, stacked
diagnostic. It is not real image FPS or deployment qualification. Native
future safety and actual-image complete-video evaluation remain necessary.
