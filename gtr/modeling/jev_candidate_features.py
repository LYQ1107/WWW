"""Canonical numeric candidate evidence; identity integers are references only."""
from dataclasses import dataclass
import torch

FEATURE_NAMES = ('gmt_score', 'proposal_margin', 'is_proposed', 'other_proposal_claims',
                 'window_observations', 'memory_observations', 'track_hits', 'view_fraction',
                 'in_active_window', 'bank_eligible', 'gallery_cosine', 'gallery_available')


@dataclass(frozen=True)
class CandidateBatch:
    candidate_ids: tuple
    scores: torch.Tensor
    state64: torch.Tensor
    evidence12: torch.Tensor
    legal_mask: torch.Tensor
    thresholds: torch.Tensor
    view: int

    def model_inputs(self):
        return self.state64, self.evidence12, self.legal_mask


def evidence12_reference(scores, ids, proposal_pairs, lengths, *, hits=None, galleries=None,
               memory_lengths=None, view_fractions=None, bank_eligible=(), observations=None):
    """Shared production/offline encoder; only observed prefix fields accepted."""
    m, n = scores.shape
    result = scores.new_zeros((m, n, 12))
    hits, galleries = hits or {}, galleries or {}
    memory_lengths, view_fractions = memory_lengths or {}, view_fractions or {}
    bank_eligible = set(bank_eligible)
    for c, track in enumerate(ids):
        gallery = galleries.get(int(track))
        available = gallery is not None and gallery.numel() > 0
        for r in range(m):
            proposed = proposal_pairs.get(r)
            baseline = scores[r, proposed] if proposed is not None else 0.
            cosine = 0.
            if available and observations is not None:
                cosine = torch.nn.functional.cosine_similarity(
                    observations[r:r+1], gallery.to(scores.device).reshape(1, -1)).item()
            result[r, c] = scores.new_tensor([
                scores[r, c].item(), (scores[r, c] - baseline).item(), float(c == proposed),
                sum(j == c for rr, j in proposal_pairs.items() if rr != r), lengths[c].item(),
                memory_lengths.get(int(track), 0), hits.get(int(track), 0),
                view_fractions.get(int(track), 0.), 1., float(int(track) in bank_eligible),
                cosine, float(available)])
    return torch.nan_to_num(result, nan=0., posinf=0., neginf=0.)


def evidence12(scores, ids, proposal_pairs, lengths, *, hits=None, galleries=None,
               memory_lengths=None, view_fractions=None, bank_eligible=(), observations=None):
    """Same canonical fields with bounded batched cosine and no per-edge sync.

    The original definition remains evidence12_reference for numeric checks.
    No state observation is manufactured: absent galleries retain unavailable
    flags and receive no cosine computation.
    """
    m, n = scores.shape
    result = scores.new_zeros((m, n, 12))
    if not m or not n:
        return result
    hits, galleries = hits or {}, galleries or {}
    memory_lengths, view_fractions = memory_lengths or {}, view_fractions or {}
    eligible = set(bank_eligible)
    columns = torch.arange(n, device=scores.device)
    rows = torch.arange(m, device=scores.device)
    proposed = torch.full((m,), -1, device=scores.device, dtype=torch.long)
    for row, col in proposal_pairs.items():
        proposed[int(row)] = int(col)
    valid = proposed >= 0
    baseline = scores.new_zeros(m)
    baseline[valid] = scores[rows[valid], proposed[valid]]
    is_proposed = columns[None, :] == proposed[:, None]
    claims = torch.bincount(proposed[valid], minlength=n)
    result[..., 0] = scores
    result[..., 1] = scores - baseline[:, None]
    result[..., 2] = is_proposed
    result[..., 3] = claims[None, :] - is_proposed.to(claims.dtype)
    result[..., 4] = torch.as_tensor(lengths, device=scores.device)
    result[..., 5] = scores.new_tensor([memory_lengths.get(int(t), 0) for t in ids])
    result[..., 6] = scores.new_tensor([hits.get(int(t), 0) for t in ids])
    result[..., 7] = scores.new_tensor([view_fractions.get(int(t), 0.) for t in ids])
    result[..., 8] = 1.
    result[..., 9] = scores.new_tensor([float(int(t) in eligible) for t in ids])
    available = [c for c, t in enumerate(ids) if galleries.get(int(t)) is not None and galleries[int(t)].numel()]
    if available:
        result[:, available, 11] = 1.
        if observations is not None:
            for start in range(0, len(available), 32):
                selected = available[start:start + 32]
                vectors = torch.stack([galleries[int(ids[c])].to(scores.device).reshape(-1) for c in selected])
                cosine = torch.nn.functional.cosine_similarity(
                    observations.to(scores.device)[:, None, :], vectors[None, :, :], dim=-1)
                result[:, selected, 10] = cosine
    return torch.nan_to_num(result, nan=0., posinf=0., neginf=0.)
