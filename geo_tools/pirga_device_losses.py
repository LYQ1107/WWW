"""Same legacy objectives with target tensors on the actual head device."""
import torch


def supervised_losses(scores,predicted,batch,supervision):
    positives=torch.tensor(supervision['certified_target_indices'],dtype=torch.long,device=scores.device)
    known=torch.tensor(supervision['candidate_known_mask'],dtype=torch.bool,device=scores.device)
    if not len(positives) or not known[positives].all():raise ValueError('Uncertified ranking targets')
    ranking=torch.logsumexp(scores[known],dim=0)-torch.logsumexp(scores[positives],dim=0)
    mask=batch.cross_view & torch.isin(batch.owners,positives)
    prediction=(1-(predicted[mask]*batch.query.detach()).sum(-1)).mean() if mask.any() else scores.sum()*0.
    return ranking,prediction,int(mask.sum())
