"""Past-history reliability diagnostic. Q2 training is gated by P0 evidence.

This head alone is NOT a deployed typed identity verifier and never selects IDs.
"""
from torch import nn

class HistoryReliabilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(24,32),nn.GELU())
        self.reliability=nn.Linear(32,3)
    def forward(self,history_features):
        return self.reliability(self.encoder(history_features))
