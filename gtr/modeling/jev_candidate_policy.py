"""Value scoring boundary shared by rules and future trained candidate models."""
import torch


class CandidateValuePolicy:
    def __init__(self, name='gmt_compat', model=None):
        if name not in ('gmt_compat', 'gmt_values', 'bidirectional', 'model'):
            raise ValueError('unsupported candidate policy')
        if name == 'model' and model is None:
            raise ValueError('candidate model must be explicitly provided')
        self.name, self.model = name, model
        self.assignment_mode = 'gmt_compat' if name == 'gmt_compat' else 'values'

    @torch.no_grad()
    def score(self, batch):
        if self.name == 'model':
            # IDs are intentionally excluded from the neural call signature.
            values, new_values = self.model(*batch.model_inputs())
        elif self.name == 'bidirectional':
            legal = batch.legal_mask & torch.isfinite(batch.scores)
            logits = batch.scores.masked_fill(~legal, -torch.inf)
            row = torch.nan_to_num(torch.softmax(logits, dim=1), nan=0.)
            col = torch.nan_to_num(torch.softmax(logits, dim=0), nan=0.)
            values = (row + col) / 2
            new_values = values.new_full((len(values),), .5)
        else:
            values = batch.scores
            if self.name == 'gmt_values':
                values = values - batch.thresholds.unsqueeze(0)
            new_values = values.new_zeros(len(values))
        if values.shape != batch.scores.shape or new_values.shape != (len(values),):
            raise ValueError('model must output values[D,C] and NEW[D]')
        return values, new_values
