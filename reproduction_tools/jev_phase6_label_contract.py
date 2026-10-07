"""Offline-only censoring of unknown identity targets; never a live policy input."""
import math

def censor_unknown_target(record):
    if record['offline_GT'] is not None or record.get('offline_unknown_target_terms_censored'):return False
    if record['sample_weight']!=0:raise AssertionError('unknown identity acquired training weight')
    for branch in record['branches'].values():
        for outcome in branch['outcomes'].values():
            # Prior rollout code could compare None == None as a recovery,
            # or call known recoveries wrong relative to an unknown target.
            # All such samples were already weight zero; correct reporting.
            outcome['utility']+=outcome.get('false_identity_recoveries',0)
            outcome['false_identity_recoveries']=0
            outcome['target_identity_recovery_latency']=None
    horizon=str(record['utility_horizon_frames'][-1]);actions=record['legal_actions']
    utilities=[record['branches'][a]['outcomes'][horizon]['utility'] for a in actions]
    masses=[math.exp(u-max(utilities)) for u in utilities]
    record['target_probs']=[m/sum(masses) for m in masses]
    record['best_actions']=[a for a,u in zip(actions,utilities) if abs(u-max(utilities))<=1e-8]
    record['offline_unknown_target_terms_censored']=True
    return True
