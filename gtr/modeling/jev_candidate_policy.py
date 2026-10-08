"""Value scoring boundary shared by rules and future trained candidate models."""
import torch


class CandidateValuePolicy:
    def __init__(self, name='gmt_compat', model=None, dynamic_alpha=0.):
        if name not in ('gmt_compat', 'gmt_values', 'bidirectional', 'model',
                        'model_fixed_new', 'dynamic_fixed_new', 'bidirectional_fixed_new'):
            raise ValueError('unsupported candidate policy')
        if name in ('model', 'model_fixed_new') and model is None:
            raise ValueError('candidate model must be explicitly provided')
        self.name, self.model = name, model
        self.dynamic_alpha = float(dynamic_alpha)
        self.assignment_mode = 'gmt_compat' if name == 'gmt_compat' else 'values'

    @torch.no_grad()
    def score(self, batch):
        if self.name in ('model', 'model_fixed_new'):
            # IDs are intentionally excluded from the neural call signature.
            values, new_values = self.model(*batch.model_inputs())
        elif self.name in ('bidirectional', 'bidirectional_fixed_new'):
            legal = batch.legal_mask & torch.isfinite(batch.scores)
            logits = batch.scores.masked_fill(~legal, -torch.inf)
            row = torch.nan_to_num(torch.softmax(logits, dim=1), nan=0.)
            col = torch.nan_to_num(torch.softmax(logits, dim=0), nan=0.)
            values = (row + col) / 2
            new_values = values.new_full((len(values),), .5)
        elif self.name == 'dynamic_fixed_new':
            e = batch.evidence12
            penalty = self.dynamic_alpha * e[..., 11] * (1 - e[..., 10])
            values = batch.scores / batch.thresholds.clamp_min(1e-6).unsqueeze(0) - 1 - penalty
            new_values = values.new_zeros(len(values))
        else:
            values = batch.scores
            if self.name == 'gmt_values':
                values = values - batch.thresholds.unsqueeze(0)
            new_values = values.new_zeros(len(values))
        if self.name.endswith('_fixed_new'):
            legal = batch.legal_mask & torch.isfinite(batch.scores)
            if values.shape[1]:
                preference_max = values.masked_fill(~legal, -torch.inf).max(1).values
                confidence = (batch.scores - batch.thresholds.unsqueeze(0)).masked_fill(~legal, -torch.inf).max(1).values
                preference_max = torch.where(legal.any(1), preference_max, torch.zeros_like(preference_max))
                confidence = torch.where(legal.any(1), confidence, torch.zeros_like(confidence))
                values = values - preference_max[:, None] + confidence[:, None]
            new_values = values.new_zeros(len(values))
        if values.shape != batch.scores.shape or new_values.shape != (len(values),):
            raise ValueError('model must output values[D,C] and NEW[D]')
        return values, new_values
