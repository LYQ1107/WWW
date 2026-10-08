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


def evidence12(scores, ids, proposal_pairs, lengths, *, hits=None, galleries=None,
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
