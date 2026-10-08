"""Capacity-matched dynamic candidate scorers with identical numeric support."""
import torch
from torch import nn


def masked_context(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n, c, d = x.shape
    if c == 0:
        return x.new_zeros((n, 2 * d))
    valid = mask.unsqueeze(-1)
    mean = (x * valid).sum(1) / valid.sum(1).clamp_min(1)
    maximum = x.masked_fill(~valid, -torch.inf).max(1).values
    maximum = torch.where(mask.any(1).unsqueeze(-1), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1)


class NormalizedCandidateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('state_mean', torch.zeros(64))
        self.register_buffer('state_std', torch.ones(64))
        self.register_buffer('evidence_mean', torch.zeros(12))
        self.register_buffer('evidence_std', torch.ones(12))
        self.register_buffer('state_constant', torch.zeros(64, dtype=torch.bool))
        self.register_buffer('evidence_constant', torch.zeros(12, dtype=torch.bool))

    def normalize(self, state: torch.Tensor, evidence: torch.Tensor,
                  mask: torch.Tensor):
        s = torch.nan_to_num((state - self.state_mean) / self.state_std)
        e = torch.nan_to_num((evidence - self.evidence_mean) / self.evidence_std)
        s = s.masked_fill(self.state_constant, 0.)
        e = e.masked_fill(self.evidence_constant, 0.).masked_fill(~mask.unsqueeze(-1), 0.)
        return s, e, masked_context(e, mask)


class CandidateMLP(NormalizedCandidateModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(100, 128), nn.ReLU(),
            nn.Linear(128, 112), nn.ReLU(), nn.Linear(112, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, state: torch.Tensor, evidence: torch.Tensor, mask: torch.Tensor):
        s, e, context = self.normalize(state, evidence, mask)
        rows, candidates = mask.shape
        x = torch.cat((s.unsqueeze(1).expand(-1, candidates, -1), e,
                       context.unsqueeze(1).expand(-1, candidates, -1)), -1)
        return self.network(x).masked_fill(~mask.unsqueeze(-1), 0.)


class CandidateDeepSets(NormalizedCandidateModel):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(88, 112), nn.ReLU())
        self.candidate_encoder = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 48), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(256, 76), nn.ReLU(), nn.Linear(76, 2))

    def forward(self, state: torch.Tensor, evidence: torch.Tensor, mask: torch.Tensor):
        s, e, context = self.normalize(state, evidence, mask)
        h = self.candidate_encoder(e).masked_fill(~mask.unsqueeze(-1), 0.)
        pooled = masked_context(h, mask)
        row = self.state_encoder(torch.cat((s, context), -1))
        c = mask.shape[1]
        x = torch.cat((row.unsqueeze(1).expand(-1, c, -1), h,
                       pooled.unsqueeze(1).expand(-1, c, -1)), -1)
        return self.scorer(x).masked_fill(~mask.unsqueeze(-1), 0.)


class CandidateJEV(NormalizedCandidateModel):
    __constants__ = ['without_qa', 'without_context']
    def __init__(self, without_qa=False, without_context=False):
        super().__init__()
        self.without_qa, self.without_context = without_qa, without_context
        self.state_encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.candidate_encoder = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 40), nn.ReLU())
        self.context_encoder = nn.Identity() if without_context else nn.Sequential(nn.Linear(24, 40), nn.ReLU())
        if without_qa:
            self.register_buffer('question', torch.zeros(8))
            self.register_buffer('action', torch.zeros(8))
        else:
            self.question = nn.Parameter(torch.zeros(8))
            self.action = nn.Parameter(torch.zeros(8))
            nn.init.normal_(self.question, std=.02)
            nn.init.normal_(self.action, std=.02)
        # Removing components also removes their inactive parameters. Widths
        # keep the functional capacity close to the three main models.
        dimension = 168 if without_context else 288
        dimension += 0 if without_qa else 16
        hidden = (132 if without_qa else 120) if without_context else (72 if without_qa else 68)
        self.scorer = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, state: torch.Tensor, evidence: torch.Tensor, mask: torch.Tensor):
        s, e, context = self.normalize(state, evidence, mask)
        h = self.candidate_encoder(e).masked_fill(~mask.unsqueeze(-1), 0.)
        row = self.state_encoder(s)
        qa = torch.cat((self.question, self.action), -1)
        n, c = mask.shape
        if self.without_context:
            x = torch.cat((row.unsqueeze(1).expand(-1, c, -1), h), -1)
        else:
            pool = masked_context(h, mask)
            context = self.context_encoder(context)
            x = torch.cat((row.unsqueeze(1).expand(-1, c, -1), h,
                context.unsqueeze(1).expand(-1, c, -1), pool.unsqueeze(1).expand(-1, c, -1)), -1)
        if not self.without_qa:
            x = torch.cat((x, qa.reshape(1, 1, 16).expand(n, c, -1)), -1)
        return self.scorer(x).masked_fill(~mask.unsqueeze(-1), 0.)


class CandidateScorer(nn.Module):
    """Frozen deployment preference; NEW is assigned by the shared policy."""
    __constants__ = ['supervision']
    def __init__(self, network, supervision='correctness', temperature=1.):
        super().__init__()
        self.network = network
        self.supervision = supervision
        self.register_buffer('temperature', torch.tensor(float(temperature)))

    def forward(self, state: torch.Tensor, evidence: torch.Tensor, mask: torch.Tensor):
        logits = self.network(state, evidence, mask)
        if self.supervision == 'H32_ranking':
            preference = logits[..., 1]
        elif self.supervision == 'joint':
            preference = logits[..., 0] + logits[..., 1]
        else:
            preference = logits[..., 0]
        return preference / self.temperature, state.new_zeros(state.shape[0])


def build_candidate_model(name):
    constructors = {'CandidateMLP': CandidateMLP, 'CandidateDeepSets': CandidateDeepSets,
        'CandidateJEV': CandidateJEV,
        'CandidateJEV_without_QA': lambda: CandidateJEV(without_qa=True),
        'CandidateJEV_without_context': lambda: CandidateJEV(without_context=True)}
    return constructors[name]()
