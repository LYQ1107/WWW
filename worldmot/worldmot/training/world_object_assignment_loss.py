"""Losses corresponding to identity, joint fusion, lifecycle and memory paths."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def world_object_assignment_loss(outputs, targets, observation_features):
    logits=outputs["assignment_logits"]
    m,n=logits.shape
    zero=sum(x.sum() for x in outputs.values())*0.0
    def tensor(value,dtype):
        return torch.as_tensor(value,device=logits.device,dtype=dtype)
    target=tensor(targets["assignment"],torch.long)
    valid=tensor(targets["observation_valid"],torch.bool)
    qvalid=tensor(targets["history_valid"],torch.bool)
    positives=tensor(targets["positive_pairs"],torch.bool)
    # Each current observation selects a historical ID or its birth alternative.
    classes=torch.cat([logits.T,outputs["birth_logits"][:,None]],dim=1)
    assignment=F.cross_entropy(classes[valid],target[valid]) if valid.any() else zero
    probabilities=classes.softmax(dim=1)
    shared=[];false_merge=[]
    for a,b in targets["same_pairs"]:
        if m and target[a]<m and target[b]<m and target[a]>=0 and target[b]>=0:
            shared.append((probabilities[a,:m]-probabilities[b,:m]).square().mean())
    for a,b in targets["different_pairs"]:
        if m:
            false_merge.append((probabilities[a,:m]*probabilities[b,:m]).sum())
    # NULL is learned against the best current assignment for this historical ID.
    if m:
        current=logits.max(dim=1).values if n else torch.zeros_like(outputs["null_logits"])
        null_loss=F.binary_cross_entropy_with_logits((outputs["null_logits"]-current)[qvalid],
            tensor(targets["null"],torch.float32)[qvalid]) if qvalid.any() else zero
        existence=F.binary_cross_entropy_with_logits(outputs["existence_logits"][qvalid],
            tensor(targets["existence"],torch.float32)[qvalid]) if qvalid.any() else zero
    else:
        null_loss,existence=zero,zero
    birth=F.binary_cross_entropy_with_logits(outputs["birth_logits"][valid],
        tensor(targets["birth"],torch.float32)[valid]) if valid.any() else zero
    pair_valid=qvalid[:,None]&valid[None,:]
    gate=F.binary_cross_entropy(outputs["update_gate"][pair_valid],positives.float()[pair_valid]) if pair_valid.any() else zero
    if positives.any():
        wanted=observation_features[None,:,:].expand(m,n,-1)[positives]
        state=(1-F.cosine_similarity(outputs["corrected_memory"][positives],wanted,dim=-1)).mean()
    else:
        state=zero
    terms={"identity_assignment":assignment,"cross_view_shared_identity":torch.stack(shared).mean() if shared else zero,
           "false_merge":torch.stack(false_merge).mean() if false_merge else zero,
           "birth":birth,"null":null_loss,"existence":existence,"update_gate":gate,
           "state_update_consistency":state}
    return {**terms,"total":sum(terms.values())}
