"""Identity and genuinely supervised cross-view birth losses, TRAIN-only targets."""
from __future__ import annotations
import torch
import torch.nn.functional as F

LOSS_WEIGHTS={"identity_assignment":1.,"joint_birth":1.,"cross_view_shared_identity":.25,
              "cross_view_identity_pair":.25,
              "false_merge":.5,"birth":1.,"null":.5,"existence":.25,
              "update_gate":.5,"state_update_consistency":.25}


def _balanced(values,positive,zero):
    parts=[]
    if positive.any():parts.append(values[positive].mean())
    if (~positive).any():parts.append(values[~positive].mean())
    return torch.stack(parts).mean() if parts else zero


def world_object_assignment_loss(outputs,targets,observation_features):
    logits=outputs["assignment_logits"];m,n=logits.shape
    zero=sum(x.sum() for x in outputs.values())*0.
    def tensor(value,dtype):return torch.as_tensor(value,device=logits.device,dtype=dtype)
    valid=tensor(targets["observation_valid"],torch.bool)
    qvalid=tensor(targets["history_valid"],torch.bool)
    positives=tensor(targets["positive_pairs"],torch.bool)
    classes=torch.cat([logits.T,outputs["birth_logits"][:,None]],dim=1)
    wanted=tensor(targets["assignment_positive"],torch.bool).T
    assignment=zero
    if valid.any():
        all_scores=torch.logsumexp(classes[valid],dim=1)
        correct=torch.logsumexp(classes[valid].masked_fill(~wanted[valid],-1e9),dim=1)
        assignment=(all_scores-correct).mean()
    probabilities=classes.softmax(dim=1)[:,:m]
    dot=probabilities@probabilities.T
    shared=zero;false_merge=zero
    if m and targets["same_pairs"]:
        edges=tensor(targets["same_pairs"],torch.long)
        same_known=positives[:,edges[:,0]].any(0)&positives[:,edges[:,1]].any(0)
        if same_known.any():
            a,b=edges[same_known].T
            shared=(probabilities[a]-probabilities[b]).square().mean()
    if m and targets["different_pairs"]:
        edges=tensor(targets["different_pairs"],torch.long);false_merge=dot[edges[:,0],edges[:,1]].mean()
    if m:
        current=logits.max(1).values if n else torch.zeros_like(outputs["null_logits"])
        null_loss=F.binary_cross_entropy_with_logits((outputs["null_logits"]-current)[qvalid],tensor(targets["null"],torch.float32)[qvalid]) if qvalid.any() else zero
        existence=F.binary_cross_entropy_with_logits(outputs["existence_logits"][qvalid],tensor(targets["existence"],torch.float32)[qvalid]) if qvalid.any() else zero
    else:null_loss,existence=zero,zero
    birth=F.binary_cross_entropy_with_logits(outputs["birth_logits"][valid],tensor(targets["birth"],torch.float32)[valid]) if valid.any() else zero
    pair_valid=qvalid[:,None]&valid[None,:]
    gate=zero
    if pair_valid.any():
        values=F.binary_cross_entropy(outputs["update_gate"][pair_valid],positives.float()[pair_valid],reduction="none")
        gate=_balanced(values,positives[pair_valid],zero)
    state=zero
    if positives.any():
        expected=observation_features[None].expand(m,n,-1)[positives]
        state=(1-F.cosine_similarity(outputs["corrected_memory"][positives],expected,dim=-1)).mean()
    joint=zero;identity_pair=zero
    if "birth_pair_logits" in outputs:
        mask=tensor(targets["joint_birth_mask"],torch.bool)
        # Model stores one canonical orientation; target storage uses index order.
        model_mask=outputs["birth_pair_mask"]|outputs["birth_pair_mask"].T
        mask=mask&model_mask
        truth=tensor(targets["joint_birth_target"],torch.float32)[mask]
        score=outputs["birth_pair_logits"][mask]
        if len(score):
            bce=F.binary_cross_entropy_with_logits(score,truth,reduction="none")
            probability=torch.sigmoid(score)
            pt=torch.where(truth>.5,probability,1-probability)
            focal=(1-pt).square()*bce
            joint=_balanced(focal,truth>.5,zero)
        identity_mask=tensor(targets["crossview_identity_mask"],torch.bool)&model_mask
        if identity_mask.any():
            identity_truth=tensor(targets["crossview_identity_target"],torch.float32)[identity_mask]
            identity_scores=outputs["birth_pair_logits"][identity_mask]
            identity_values=F.binary_cross_entropy_with_logits(identity_scores,identity_truth,reduction="none")
            identity_pair=_balanced(identity_values,identity_truth>.5,zero)
    terms={"identity_assignment":assignment,"joint_birth":joint,"cross_view_shared_identity":shared,
           "cross_view_identity_pair":identity_pair,
           "false_merge":false_merge,"birth":birth,"null":null_loss,"existence":existence,
           "update_gate":gate,"state_update_consistency":state}
    return {**terms,"total":sum(LOSS_WEIGHTS[k]*value for k,value in terms.items())}
