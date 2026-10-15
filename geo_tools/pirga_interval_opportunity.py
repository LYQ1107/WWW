"""Offline interval feasibility diagnostics, never a deployable GT policy.

Allow arbitrary per-edge residuals within frozen model bounds, a superset of
learned-model expressivity. A strict interval objective exclusion proves some
repairs impossible; absence of exclusion does not prove learnability. Witnesses
are GT-assisted diagnostic matrices and must never be counted as learned gain.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment(a):
    rr,cc=linear_sum_assignment(-a);return rr,cc,float(a[rr,cc].sum(dtype=np.float64))


def forced(a,row,col):
    other=np.delete(np.delete(a,row,axis=0),col,axis=1);rr,cc,value=assignment(other)
    rows=[v for v in range(a.shape[0]) if v!=row];cols=[v for v in range(a.shape[1]) if v!=col]
    edges=[(row,col)]+[(rows[int(r)],cols[int(c)]) for r,c in zip(rr,cc)]
    return value+float(a[row,col]),edges


def interval_opportunity(base,caps,thresholds,row,positives):
    base=np.asarray(base,dtype=np.float64);caps=np.asarray(caps,dtype=np.float64);thresholds=np.asarray(thresholds,dtype=np.float64)
    if base.shape!=caps.shape or (caps<0).any() or not np.isfinite(base).all() or not np.isfinite(caps).all() or len(thresholds)!=base.shape[1]:raise ValueError('Full finite interval matrix required')
    positives=list(positives);upper=base+caps;lower=base-caps
    viable=[j for j in positives if upper[row,j]>thresholds[j]]
    if not viable:return dict(certified_impossible=True,reason='ALL_PURE_POSITIVES_BELOW_STRICT_THRESHOLD_EVEN_AT_UPPER_BOUND',interval_witness_found=False)
    good_upper=max(forced(upper,row,j)[0] for j in viable)
    bad_lower=lower.copy();bad_lower[row,positives]=-np.inf
    try:bad_value=assignment(bad_lower)[2]
    except ValueError:bad_value=None
    excluded=bad_value is not None and good_upper<bad_value-1e-6
    witnessed=False
    for col in viable:
        _,edges=forced(upper,row,col);witness=lower.copy()
        for r,c in edges:witness[r,c]=upper[r,c]
        rr,cc,_=assignment(witness)
        selected=[int(c) for r,c in zip(rr,cc) if int(r)==row]
        if selected and selected[0] in positives and witness[row,selected[0]]>thresholds[selected[0]]:witnessed=True;break
    if excluded and witnessed:raise ValueError('Contradictory interval exclusion and witness')
    return dict(certified_impossible=excluded,reason='FULL_JOINT_INTERVAL_OBJECTIVE_EXCLUDES_PURE_POSITIVE' if excluded else 'NOT_EXCLUDED_BY_NECESSARY_TEST',
        best_forced_positive_upper_objective=good_upper,best_no_positive_lower_objective=bad_value,interval_witness_found=witnessed,
        witness_qualification='GT-assisted arbitrary bounded edge changes only; not learned policy, not online success')
