"""Separate compatibility and commitment certificates, UNKNOWN-safe joint loss."""
import torch
from torch.nn import functional as F
from jev_phase14_losses import safe_joint_loss, objective as old_objective


def mass_loss(logits, positive, known, valid):
    lp = F.log_softmax(logits.masked_fill(~known, -1e4), -1)
    mass = torch.logsumexp(lp.masked_fill(~positive, -1e4), -1)
    return -mass[valid].mean() if valid.any() else logits.sum()*0


def binary(logits, targets, mask):
    selected = mask & (targets >= 0)
    return F.binary_cross_entropy_with_logits(logits[selected], targets[selected].float()) if selected.any() else logits.sum()*0


def objective(details, x, y, arm):
    if arm == 'A_original': return old_objective(details, x, y, 'availability_joint')
    who = mass_loss(details['choice_logits'], y['positive'], y['known_options'], y['supervised'] & x['question_mask'])
    availability = binary(details['availability_logits'], y['availability'], x['question_mask'])
    trust = binary(details['trust_logits'], y['trust'], x['legal'])
    valid = (y['commit_kind'] > 0) & x['question_mask']
    if arm == 'B_continuity': valid = (y['commit_kind'] == 1) & x['question_mask']
    commit = mass_loss(details.get('commitment_logits', details['logits']), y['commit_positive'], y['known_options'], valid)
    target = dict(y)
    if arm in ['C_commitment', 'D_fixed', 'E_set', 'F_full']:
        target['positive'] = torch.where(valid[..., None], y['commit_positive'], y['positive'])
    assignment = safe_joint_loss(details['logits'], x, target)
    risk = details['logits'].sum()*0
    if 'purity_logits' in details:
        risk = binary(details['purity_logits'], y['trust'], x['legal']) + binary(
            details['safety_logits'], y['safety'], x['legal']) + binary(
            details['uncertainty_logits'], y['uncertainty'], x['legal'])
    total = who + .5*availability + .5*trust + commit + .2*assignment + .5*risk
    return total, dict(WHO=float(who.detach()), Availability=float(availability.detach()), Trust=float(trust.detach()),
                      Commitment=float(commit.detach()), JointAssignment=float(assignment.detach()), UnsafeSwitchRisk=float(risk.detach()))
