# Complete Guard development result

Guard completed all 12 registered DEV videos (30,702 images). The other controls remain incomplete after the registered margin_video2 one-hour timeout; the timeout and all costs are retained separately.

Unmodified evaluators can use only TRAIN01/10/11/18/19 because the other seven videos contain invalid duplicated raw GT. Those seven remain in complete inference and offline owner diagnostics; their strict tracking metrics remain null. GT was not repaired and never entered inference.

With original stock50 postprocessing, Guard HOTA is 81.0397601 versus Original 81.0550722 (−0.0153120); AssA 81.9927441 versus 81.8954951; CVIDF1 95.6626979 versus 95.6316643; IDSW 46 versus 41. Relative to C1, stock HOTA increases only 0.0034862 and IDSW falls from48 to46. Raw0 gives HOTA80.8250083 and IDSW364, but the raw gain does not establish improvement under the original postprocessing.

Actual full-image FPS over all12 is4.4159151 (Original6.2360430; C16.1671106). The implementation includes full state cloning on surviving interventions. Prefix COW engineering qualification cannot be substituted for these measured times.

Offline strictly-past owner certification finds5618 wrong→correct and5387 correct→wrong observations in50 repaired and61 harmed scene/person dependence groups. These counts include the underlying C1 changes and are not independent Guard successes. UNKNOWN/unanchored cases are separate; groups overlap. This mixed result does not establish a safe identity improvement or qualify postcommit SPLIT/RELINK.

The separate engineering stage passed all seven actual-image probes, including full/COW raw predictions, native states, RNG, solver costs and both frozen-head fault recoveries. Its actual cost was0.07614154 GPU-wall-hours. This qualifies only the40-group engineering scope, not full videos, production performance, learned policy efficacy or TEST.

Next: continue the registered learned/reference controls after their own native feature/logit, fault, OFF and full-video admission. Preserve the incomplete rule comparisons and do not change thresholds or choose favorable scenes.
