"""GT-free original-descriptor C1; native candidate columns are never created.

The caller supplies actual native history descriptor versions. Bank fallback
keeps its original separate scoring path in this first C1 integration.
"""
import torch
import torch.nn.functional as F


def c1_scores(native_scores,query_features,history_features,history_ids,
              candidate_ids,history_frames,query_frame,alpha=2.0):
    if native_scores.shape != (len(query_features),len(candidate_ids)):
        raise ValueError('CIR query/candidate matrix contract')
    if query_features.ndim!=2 or query_features.shape[1]!=1152 or history_features.shape!=(len(history_ids),1152):
        raise ValueError('CIR requires actual full 1152-dimensional GMT descriptors')
    if history_frames.shape!=(len(history_ids),):raise ValueError('History source/frame contract')
    if len(candidate_ids)!=len(torch.unique(candidate_ids)):raise ValueError('Duplicate native candidate column')
    if not torch.isfinite(query_features).all() or not torch.isfinite(history_features).all() or not torch.isfinite(native_scores).all():
        raise ValueError('Nonfinite native evidence')
    if (history_frames>query_frame).any():raise ValueError('Future evidence rejected')
    if len(history_ids) and not (history_ids[:,None]==candidate_ids[None,:]).any(1).all():
        raise ValueError('History names a nonnative identity')
    keep=history_frames<query_frame
    # Earlier current-group cameras remain native GTA candidates, but are not
    # promoted to strict-past C1 evidence. An unsupported candidate gets delta0.
    h=F.normalize(history_features[keep].float(),dim=-1)
    owners=(history_ids[keep,None]==candidate_ids[None,:]).to(torch.float32)
    prototypes=F.normalize(owners.T@h,dim=-1)
    q=F.normalize(query_features.float(),dim=-1)
    return native_scores+float(alpha)*(q@prototypes.T)
