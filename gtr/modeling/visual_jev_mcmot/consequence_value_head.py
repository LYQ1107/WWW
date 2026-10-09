from torch import nn

class ConsequenceValueHead(nn.Module):
    """Predict from current action evidence; true future values are not inputs."""
    def __init__(self,d=128):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,3))
    def forward(self,actions):
        return self.network(actions)
