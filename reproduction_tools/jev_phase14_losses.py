"""UNKNOWN-safe cost augmentation admits all lawful unknown alternatives at zero cost."""
import torch
from torch.nn import functional as F
from gtr.modeling.jev_stage2.assignment import lawful_choice,structured_assignment_loss

def safe_joint_loss(logits,x,y,cost_sensitive=True):
    terms=[];positive=y['positive'];known=y['known_options']
    for b in range(len(logits)):
        rows=torch.where(y['supervised'][b]&x['question_mask'][b])[0]
        if not len(rows):continue
        z=logits[b,rows];legal=x['legal'][b,rows];k=legal.shape[-1]
        allowed=positive[b,rows] | ~known[b,rows]
        full_legal=torch.cat([legal,torch.ones(len(rows),1,device=z.device,dtype=torch.bool)],1)
        allowed &= full_legal
        # All legal unknown options remain admissible. No gradient punishes their
        # selection merely because they lack a GT certificate.
        costs=(known[b,rows]&~positive[b,rows]).float()*(2. if cost_sensitive else 1.)
        costs=costs.masked_fill(~full_legal,0)
        oracle_legal=legal & allowed[:,:k]
        oracle=z.clone();oracle[:,-1]=oracle[:,-1].masked_fill(~allowed[:,-1],-1e9)
        base=lawful_choice(oracle,oracle_legal)
        if any(c<0 and not bool(allowed[r,-1]) for r,c in enumerate(base)):
            # Contradictory certified per-camera capacity: do not manufacture a label.
            continue
        augmented=lawful_choice(z+costs,legal)
        idx=torch.arange(len(rows),device=z.device)
        first=torch.tensor([k if c<0 else c for c in base],device=z.device)
        second=torch.tensor([k if c<0 else c for c in augmented],device=z.device)
        terms.append(torch.relu((z[idx,second]+costs[idx,second]).sum()-z[idx,first].sum())/len(rows))
    return torch.stack(terms).mean() if terms else logits.sum()*0

def objective(details,x,y,kind):
    z=details['logits'];known=y['known_options'];positive=y['positive'];valid=y['supervised']&x['question_mask']
    logp=F.log_softmax(z.masked_fill(~known,-1e4),-1)
    mass=torch.logsumexp(logp.masked_fill(~positive,-1e4),-1)
    ce=-mass[valid].mean() if valid.any() else z.sum()*0
    target=positive.float()/positive.sum(-1,keepdim=True).clamp_min(1)
    brier=(((logp.exp()-target)**2*known).sum(-1)[valid]).mean() if valid.any() else z.sum()*0
    structure=z.sum()*0
    if kind=='legacy_structured':structure=structured_assignment_loss(z,x['legal']&known[:,:,:-1],y['targets'],valid)
    elif kind in ['cost_sensitive','availability_joint']:structure=safe_joint_loss(z,x,y)
    avail_known=(y['availability']>=0)&x['question_mask']
    availability=F.binary_cross_entropy_with_logits(details['availability_logits'][avail_known],y['availability'][avail_known].float()) if avail_known.any() else z.sum()*0
    trust_known=(y['trust']>=0)&x['legal']
    trust=F.binary_cross_entropy_with_logits(details['trust_logits'][trust_known],y['trust'][trust_known].float()) if trust_known.any() else z.sum()*0
    total=ce
    if kind!='standard_choice':total=total+.25*structure+.1*brier
    if kind=='availability_joint':total=total+.5*availability+.25*trust
    return total,dict(CE=float(ce.detach()),structure=float(structure.detach()),Brier=float(brier.detach()),
                     availability=float(availability.detach()),trust=float(trust.detach()),known_rows=int(valid.sum()))
