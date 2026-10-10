"""Separate visual WHO compatibility from a concrete persistent ID commitment."""
import torch
from torch import nn
from torch.nn import functional as F
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from .commitment_question import CommitmentQuestion
from .commitment_option_reader import CommitmentOptionReader
from .candidate_reliability import CandidateReliability

ARMS = ['A_original', 'B_continuity', 'C_commitment', 'D_fixed', 'E_set', 'F_full']


class PersistentIdentityPolicy(nn.Module):
    def __init__(self, arm='F_full', ablation=None):
        super().__init__(); assert arm in ARMS; self.arm = arm; self.ablation = ablation
        variant = 'fixed_question' if arm == 'D_fixed' else 'set_transformer' if arm == 'E_set' else 'multi_question'
        self.identity = ReliableIdentityPolicy(variant, 'availability_joint')
        if arm in ARMS[2:]:
            question = 'fixed' if arm == 'D_fixed' or ablation == 'fixed_Q4' else 'pooled' if arm == 'E_set' else 'dynamic'
            option = 'ordinary' if ablation == 'ordinary_option' else 'fixed_control' if question == 'fixed' else 'set' if arm == 'E_set' else 'dynamic'
            self.question = CommitmentQuestion(question)
            self.option = CommitmentOptionReader(option)
        if arm in ['D_fixed', 'E_set', 'F_full']: self.reliability = CandidateReliability()

    def details(self, x):
        assert 'commitment_features' in x
        base = {key: val for key, val in x.items() if key != 'commitment_features'}
        result = self.identity.details(base)
        z = result['logits']
        if x['question_mask'].shape[1] == 0: return result
        if self.arm in ARMS[2:]:
            query, tokens, det = self.question(x, result['choice_logits'], result['trust_logits'], result['availability_logits'])
            value = self.option(x, query, tokens, det)
            result['commitment_logits'] = z + value
            z = z + value
        if hasattr(self, 'reliability'):
            risk = self.reliability(x); result.update(risk)
            # Purity alone never certifies identity compatibility.
            adjustment = F.logsigmoid(risk['safety_logits']) + F.logsigmoid(-risk['uncertainty_logits']) + 1.3862943611198906
            z = torch.cat([z[:, :, :-1] + adjustment, z[:, :, -1:]], -1)
        legal = torch.cat([x['legal'], torch.ones_like(x['question_mask'])[:, :, None]], -1)
        result['logits'] = z.masked_fill(~legal, -1e4)
        return result

    def forward(self, x):
        details = self.details(x)
        if not self.training:
            self.runtime_details = {key: value.detach() for key, value in details.items()}
        return details['logits']
