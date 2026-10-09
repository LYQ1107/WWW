"""Lawful native assignment. DEFER is not the old bank-veto NEW flag."""
import numpy as np
import torch
from ..jev_candidate_assignment import assign_candidate_values,CandidateAssignment,array

def conflict_components(legal,views=None):
    edges=array(legal).astype(bool)
    m,n=edges.shape
    scopes=np.zeros(m,dtype=int) if views is None else array(views)
    unseen=set(range(m));groups=[]
    while unseen:
        root=min(unseen);unseen.remove(root);group={root};columns=set(np.flatnonzero(edges[root]))
        changed=True
        while changed:
            changed=False
            for r in sorted(unseen.copy()):
                if scopes[r]==scopes[root] and columns.intersection(np.flatnonzero(edges[r])):
                    group.add(r);unseen.remove(r);columns.update(np.flatnonzero(edges[r]));changed=True
        groups.append(tuple(sorted(group)))
    return groups

def lawful_match(values,batch,abstain=None,views=None):
    base=assign_candidate_values(batch.scores,batch.scores.new_zeros(len(batch.scores)),batch.candidate_ids,batch.legal_mask,views=views,mode='gmt_compat',legacy_thresholds=batch.thresholds)
    choice=assign_candidate_values(values,values.new_zeros(len(values)),batch.candidate_ids,batch.legal_mask,views=views)
    result=list(choice.existing_ids);pairs=dict(choice.pairs)
    fallback=[]
    if abstain is not None:
        risky=array(abstain).astype(bool)
        original_pairs=dict(base.pairs)
        for group in conflict_components(batch.legal_mask,views):
            if any(risky[r] for r in group):
                fallback.extend(group)
                for r in group:
                    result[r]=base.existing_ids[r];pairs.pop(r,None)
                    if r in original_pairs:pairs[r]=original_pairs[r]
    # Components sharing any legal ID were solved together; mixed policy groups
    # therefore cannot allocate the same ID to two rows in one camera.
    scopes=np.zeros(len(result),dtype=int) if views is None else array(views)
    for camera in set(scopes.tolist()):
        ids=[x for r,x in enumerate(result) if x>=0 and scopes[r]==camera]
        if len(ids)!=len(set(ids)):raise AssertionError('per-camera capacity violated')
    return CandidateAssignment(tuple(result),tuple(r for r,x in enumerate(result) if x<0),tuple(sorted(pairs.items())),False),tuple(sorted(fallback))

def transition_request(action):
    """No ID allocation here; birth belongs exclusively to final native stage."""
    from .schemas import ActionType
    if action==ActionType.ABSTAIN_OR_FALLBACK:return {'request':'GROUP_NATIVE_FALLBACK','birth_increment':0,'bank_query':False}
    if action==ActionType.DEFER_TO_REACTIVATION:return {'request':'UNMATCHED_BANK_QUERY','birth_increment':0,'bank_query':True}
    if action==ActionType.START_NEW:return {'request':'FINAL_NATIVE_BIRTH','birth_increment':1,'bank_query':False}
    raise ValueError('not a special transition request')
