"""Membership and identity contrast only; Ω estimates are never 3D GT targets."""
import torch
from torch.nn import functional as F

LOSS_WEIGHTS={"per_camera_membership":1.,"detection_exclusivity":.5,"hard_negative_membership":.25,
              "cross_view_membership":.25,"cross_view_visual_contrast":.25}


def membership_loss(outputs,targets):
    score=outputs["membership_logits"];null=outputs["no_observation_logits"];m,n=score.shape
    zero=score.sum()*0+null.sum()*0+outputs["projected_observations"].sum()*0
    device=score.device
    tensor=lambda x,dtype:torch.as_tensor(x,device=device,dtype=dtype)
    positive=tensor(targets["positive"],torch.bool);qvalid=tensor(targets["history_valid"],torch.bool)
    valid=tensor(targets["observation_valid"],torch.bool);camera=tensor(targets["camera_slots"],torch.long)
    row_losses=[]
    for view in range(2):
        mask=valid&(camera==view)
        if not bool(qvalid.any()):continue
        current=score[qvalid][:,mask];p=positive[qvalid][:,mask]
        classes=torch.cat([current,null[qvalid,view,None]],dim=1)
        correct=torch.cat([p,~p.any(1,keepdim=True)],dim=1)
        row_losses.append((torch.logsumexp(classes,1)-torch.logsumexp(classes.masked_fill(~correct,-1e9),1)).mean())
    per_camera=torch.stack(row_losses).mean() if row_losses else zero
    exclusive=zero;prob=score.new_zeros((m,n))
    if n:
        # Constant UNMATCHED is an association option, never a learned birth
        # classifier. All actual unmatched detections reach the birth manager.
        classes=torch.cat([score.masked_fill(~qvalid[:,None],-1e9),score.new_zeros((1,n))],dim=0)
        wanted=torch.cat([positive&qvalid[:,None],~(positive&qvalid[:,None]).any(0,keepdim=True)],dim=0)
        if bool(valid.any()):exclusive=(torch.logsumexp(classes[:,valid],0)-torch.logsumexp(classes[:,valid].masked_fill(~wanted[:,valid],-1e9),0)).mean()
        prob=classes.softmax(0)[:m]
    pairmask=qvalid[:,None]&valid[None,:];positive_mask=pairmask&positive;negative_mask=pairmask&~positive
    hard=[]
    if bool(positive_mask.any()):hard.append(F.softplus(-score[positive_mask]).mean())
    if bool(negative_mask.any()):
        negatives=score.masked_fill(~negative_mask,-1e9)
        hardest=negatives.topk(min(4,m),dim=0).values
        use=hardest>-1e8
        if bool(use.any()):hard.append(F.softplus(hardest[use]).mean())
    hard_negative=torch.stack(hard).mean() if hard else zero
    same=[];different=[];labels=targets["labels"]
    for a in range(n):
        if labels[a] is None:continue
        for b in range(a+1,n):
            if labels[b] is None or targets["camera_slots"][a]==targets["camera_slots"][b]:continue
            if labels[a]==labels[b]:same.append((a,b))
            else:different.append((a,b))
    shared=[]
    if m and same:
        pairs=tensor(same,torch.long);a,b=pairs.T;known=(positive[:,a]&positive[:,b]&qvalid[:,None]).any(0)
        if bool(known.any()):
            a,b=a[known],b[known];correct=(positive[:,a]&positive[:,b]&qvalid[:,None]).float()
            shared.append(-torch.log((prob[:,a]*prob[:,b]*correct).sum(0).clamp_min(1e-7)).mean())
    if m and different:
        pairs=tensor(different,torch.long);a,b=pairs.T
        shared.append((prob[:,a]*prob[:,b]).sum(0).mean())
    cross=torch.stack(shared).mean() if shared else zero
    visual=[];embedding=outputs["projected_observations"]
    for source,target in ((0,1),(1,0)):
        a=[j for j in range(n) if targets["camera_slots"][j]==source and labels[j] is not None]
        b=[j for j in range(n) if targets["camera_slots"][j]==target and labels[j] is not None]
        if not a or not b:continue
        logits=embedding[a]@embedding[b].T/.07
        same_label=tensor([[labels[i]==labels[j] for j in b] for i in a],torch.bool);known=same_label.any(1)
        if bool(known.any()):visual.append((torch.logsumexp(logits[known],1)-torch.logsumexp(logits[known].masked_fill(~same_label[known],-1e9),1)).mean())
    contrast=torch.stack(visual).mean() if visual else zero
    terms={"per_camera_membership":per_camera,"detection_exclusivity":exclusive,"hard_negative_membership":hard_negative,
           "cross_view_membership":cross,"cross_view_visual_contrast":contrast}
    return {**terms,"total":sum(LOSS_WEIGHTS[k]*v for k,v in terms.items())}

