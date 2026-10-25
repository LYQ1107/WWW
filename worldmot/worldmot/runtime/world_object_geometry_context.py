"""Attach current alignment evidence only to registered C4/C5 observations."""
from copy import copy
import numpy as np


def with_alignment_context(observations,alignment):
    result=[]
    confidence=1. if alignment.get('reason')=='initial_reference' else alignment.get('inlier_ratio')
    if confidence is None:confidence=1. if alignment.get('success') else 0.
    confidence=float(np.clip(confidence,0,1)) if np.isfinite(confidence) else 0.
    if not alignment.get('success'):confidence=0.
    residual=alignment.get('residual_p95')
    if residual is not None and not np.isfinite(residual):residual=None
    for original in observations:
        o=copy(original)
        o.source={**o.source,'alignment_confidence':confidence,
                  'alignment_residual_p95':residual,'alignment_evidence_source':'CURRENT_ALIGNMENT'}
        result.append(o)
    return result
