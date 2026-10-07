"""Same one-row B2 operators with Python dispatch removed, no new weights."""
import torch
from torch import nn


class FixedLegalLogits(nn.Module):
    def __init__(self,controller,legal):
        super().__init__();self.controller=controller;self.legal=legal

    def forward(self,features):
        return self.controller(features,['MATCH_DECISION'],[self.legal])['logits']


class TracedMatchPolicy:
    def __init__(self,controller):
        self.controller=controller
        self.graphs={}
        for legal in (('ACCEPT_CURRENT','REASSOCIATE','START_NEW'),('ACCEPT_CURRENT','START_NEW')):
            wrapped=FixedLegalLogits(controller,list(legal)).eval()
            self.graphs[legal]=torch.jit.trace(wrapped,torch.zeros(1,64),check_trace=True,strict=True)

    @torch.no_grad()
    def action(self,features,legal):
        if legal==['START_NEW']:return 'START_NEW'
        logits=self.graphs[tuple(legal)](features.detach().cpu().reshape(1,64))
        return legal[int(logits[0].argmax())]


class FastMatchPolicy:
    """Cache fixed semantic tokens while retaining every eager float operation."""
    def __init__(self,controller):
        from gtr.modeling.jev_decision import ACTION_TO_INDEX
        if controller.option_interaction is not None:
            raise ValueError('cached semantic keys require independent frozen option scoring')
        self.controller=controller;self.keys={}
        with torch.no_grad():
            self.question=controller.question_embedding(torch.tensor([0]))
            for legal in (('ACCEPT_CURRENT','REASSOCIATE','START_NEW'),('ACCEPT_CURRENT','START_NEW')):
                ids=torch.tensor([[ACTION_TO_INDEX[a] for a in legal]])
                self.keys[legal]=controller.key(controller.action_embedding(ids))

    @torch.no_grad()
    def logits(self,features,legal):
        x=features.detach().cpu().float().reshape(1,64)
        state=self.controller.state_encoder(x)
        query=self.controller.query(torch.cat([state,self.question],dim=-1))
        keys=self.keys[tuple(legal)]
        logits=torch.sum(query.unsqueeze(1)*keys,dim=-1)
        return logits/(self.controller.temperature*(keys.shape[-1]**0.5))

    @torch.no_grad()
    def action(self,features,legal):
        if legal==['START_NEW']:return 'START_NEW'
        # Preserve eager probability rounding and its legal-order tie break.
        probs=torch.softmax(self.logits(features,legal),dim=-1)
        return legal[int(probs[0].argmax())]
