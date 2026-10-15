"""Exact per-call reuse of query-independent historical projections on CPU.

This changes neither evidence nor scores. No persistent feature cache is made.
Mixed-camera/time calls fall back to the original per-query implementation.
"""
import torch
import torch.nn.functional as F

from cv_pil_v2_model import tensorize, fused_scores

ARMS = ('C0','C1','C2','C3','C4')


def call_scores(assets, call, models):
    queries = call['queries']
    if not queries:
        return {arm: call['track_scores'].clone() for arm in ARMS}
    metadata = [assets.observations[tuple(q)] for q in queries]
    same_context = len({(o['frame'],o['camera'],o['image_id']) for o in metadata}) == 1
    if not same_context:
        output={arm:[] for arm in ARMS}
        for source in queries:
            b=tensorize(assets.actor((12,*source)))
            scores=dict(C0=b.native_scores,C1=fused_scores(b,b.histories))
            scores.update({arm:model(b)[0] for arm,model in models.items()})
            for arm in ARMS:output[arm].append(scores[arm])
        return {arm:torch.stack(values) for arm,values in output.items()}
    first=tensorize(assets.actor((12,*queries[0])))
    projected={'C1':first.histories}
    projected.update({arm:model.project(first) for arm,model in models.items()})
    prototypes={}
    for arm,h in projected.items():
        p=torch.zeros(len(first.native_scores),1152,dtype=h.dtype,device=h.device)
        p=p.index_add(0,first.owners,h)
        prototypes[arm]=F.normalize(p,dim=-1)
    output={arm:[] for arm in ARMS}
    for row,source in enumerate(queries):
        query=F.normalize(assets.feature(call['query_feature_keys'][row]).detach().float().clone(),dim=0)
        native=call['track_scores'][row].detach().float().clone()
        output['C0'].append(native)
        for arm in ('C1','C2','C3','C4'):
            output[arm].append(native+2.*(prototypes[arm]@query))
    return {arm:torch.stack(values) for arm,values in output.items()}
