"""Preregistered normalization-matched generic MLP, no question/action embeddings."""
from jev_phase7_common import OUT
from torch import nn
from jev_phase7_policies import StrongMLP

class NormalizedMLP(StrongMLP):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(64,152),nn.LayerNorm(152),nn.GELU(),
                                   nn.Linear(152,152),nn.LayerNorm(152),nn.GELU(),nn.Linear(152,3))

if __name__=='__main__':
    from jev_phase7_common import OUT
    import fit_jev_phase7_controls as fitting
    fitting.OUT=OUT/'normalization_control'
    fitting.StrongMLP=NormalizedMLP
    fitting.train('MLP','cuda:0')
