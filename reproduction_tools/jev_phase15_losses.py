"""Separate compatibility and commitment certificates, UNKNOWN-safe joint loss."""
import torch
from torch.nn import functional as F
from jev_phase14_losses import safe_joint_loss, objective as old_objective
from gtr.modeling.jev_stage2.assignment import lawful_choice


def whole_payload_joint_loss(logits,x,y):
    terms=[]
    for b in range(len(logits)):
        rows=torch.where(x['question_mask'][b])[0]
        if not len(rows):continue
        z=logits[b,rows];legal=x['legal'][b,rows];k=legal.shape[-1]
        supervised=y['supervised'][b,rows]
        known=y['known_options'][b,rows]&supervised[:,None]
        positive=y['positive'][b,rows]
        admissible=positive|~known
        full_legal=torch.cat([legal,torch.ones(len(rows),1,dtype=torch.bool,device=z.device)],1)
        admissible &= full_legal
        costs=2*(known&~positive).float()
        oracle=z.clone();oracle[:,-1]=oracle[:,-1].masked_fill(~admissible[:,-1],-1e9)
        first=lawful_choice(oracle,legal&admissible[:,:k])
        if any(c<0 and not admissible[row,-1] for row,c in enumerate(first)):continue
        second=lawful_choice(z+costs,legal)
        idx=torch.arange(len(rows),device=z.device)
        a=torch.tensor([k if c<0 else c for c in first],device=z.device)
        c=torch.tensor([k if choice<0 else choice for choice in second],device=z.device)
        terms.append(torch.relu((z[idx,c]+costs[idx,c]).sum()-z[idx,a].sum())/len(rows))
    return torch.stack(terms).mean() if terms else logits.sum()*0


def mass_loss(logits, positive, known, valid):
    lp = F.log_softmax(logits.masked_fill(~known, -1e4), -1)
    mass = torch.logsumexp(lp.masked_fill(~positive, -1e4), -1)
    return -mass[valid].mean() if valid.any() else logits.sum()*0


def binary(logits, targets, mask):
    selected = mask & (targets >= 0)
    return F.binary_cross_entropy_with_logits(logits[selected], targets[selected].float()) if selected.any() else logits.sum()*0


def objective(details, x, y, arm):
    if arm == 'A_original': return old_objective(details, x, y, 'availability_joint')
    if arm == 'B_continuity':
        baseline,parts=old_objective(details,x,y,'availability_joint')
        valid=(y['commit_kind']==1)&x['question_mask']
        continuity=mass_loss(details['logits'],y['commit_positive'],y['known_options'],valid)
        return baseline+continuity,dict(parts,ordinary_continuity=float(continuity.detach()))
    who = mass_loss(details['choice_logits'], y['positive'], y['known_options'], y['supervised'] & x['question_mask'])
    availability = binary(details['availability_logits'], y['availability'], x['question_mask'])
    trust = binary(details['trust_logits'], y['trust'], x['legal'])
    valid = (y['commit_kind'] > 0) & x['question_mask']
    commit = mass_loss(details.get('commitment_logits', details['logits']), y['commit_positive'], y.get('commit_known_options',y['known_options']), valid)
    target = dict(y)
    if arm in ['C_commitment', 'D_fixed', 'E_set', 'F_full']:
        target['positive'] = torch.where(valid[..., None], y['commit_positive'], y['positive'])
        target['known_options'] = torch.where(valid[...,None], y.get('commit_known_options',y['known_options']),y['known_options'])
    assignment = whole_payload_joint_loss(details['logits'], x, target)
    risk = details['logits'].sum()*0
    if 'purity_logits' in details:
        risk = binary(details['purity_logits'], y['trust'], x['legal']) + binary(
            details['safety_logits'], y['safety'], x['legal']) + binary(
            details['uncertainty_logits'], y['uncertainty'], x['legal'])
    total = who + .5*availability + .5*trust + commit + .2*assignment + .5*risk
    return total, dict(WHO=float(who.detach()), Availability=float(availability.detach()), Trust=float(trust.detach()),
                      Commitment=float(commit.detach()), JointAssignment=float(assignment.detach()), UnsafeSwitchRisk=float(risk.detach()))
