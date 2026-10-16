"""Small native risk classifiers; no GT, identities or future-model interface.

Both registered architectures have961 actual learnable parameters. These
logits classify a selected edge's risk, not the causal utility of vetoing it.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn


INPUT_CONTRACT='cir-native-risk12-v1'
GLOBAL_INDICES=(0,1,2,8,9,11)
LOCAL_INDICES=(3,4,5,6,7,10)


class NativeRiskHead(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=architecture
        self.register_buffer('feature_mean',torch.zeros(12,device='cpu',dtype=torch.float32))
        self.register_buffer('feature_scale',torch.ones(12,device='cpu',dtype=torch.float32))
        factory=dict(device='cpu',dtype=torch.float32)
        if architecture=='ordinary_mlp':
            self.network=nn.Sequential(nn.Linear(12,32,**factory),nn.ReLU(),nn.Linear(32,16,**factory),nn.ReLU(),nn.Linear(16,1,**factory))
        elif architecture=='grouped_risk':
            self.register_buffer('token_indices',torch.tensor([GLOBAL_INDICES,LOCAL_INDICES],device='cpu',dtype=torch.long))
            self.encoder=nn.Linear(6,8,**factory)
            self.type_bias=nn.Parameter(torch.zeros(2,8,**factory))
            self.pool_query=nn.Parameter(torch.zeros(8,**factory))
            self.network=nn.Sequential(nn.Linear(8,88,**factory),nn.ReLU(),nn.Linear(88,1,**factory))
        else:raise ValueError('Unregistered risk architecture')
        if sum(p.numel() for p in self.parameters())!=961:raise RuntimeError('Matched parameter capacity drift')

    def set_normalization(self,mean,scale):
        if mean.shape!=(12,) or scale.shape!=(12,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<.001).any():
            raise ValueError('Frozen finite native-risk12 normalization required')
        self.feature_mean.copy_(mean);self.feature_scale.copy_(scale)

    def forward(self,features):
        if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=12 or not features.is_floating_point():
            raise ValueError('Only an N x12 floating native evidence tensor is accepted')
        if not torch.isfinite(features).all():raise ValueError('Nonfinite native risk evidence')
        x=(features.float()-self.feature_mean)/self.feature_scale
        if self.architecture=='grouped_risk':
            tokens=torch.relu(self.encoder(x[:,self.token_indices])+self.type_bias)
            weight=torch.softmax((tokens*self.pool_query).sum(-1)/math.sqrt(8),dim=1)
            x=(tokens*weight[:,:,None]).sum(1)
        return self.network(x).squeeze(-1)


def load_native_risk(path,expected_sha256,device='cpu'):
    """Verified independent checkpoint, without consuming deployment RNG."""
    path=Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Risk checkpoint SHA mismatch')
    with torch.random.fork_rng(devices=[]):
        checkpoint=torch.load(path,map_location='cpu')
        if checkpoint['input_contract']!=INPUT_CONTRACT:raise ValueError('Risk input contract mismatch')
        threshold=float(checkpoint['threshold'])
        if not math.isfinite(threshold):raise ValueError('Finite frozen deployment threshold required')
        model=NativeRiskHead(checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        if not torch.isfinite(model.feature_mean).all() or not torch.isfinite(model.feature_scale).all() or (model.feature_scale<.001).any():
            raise ValueError('Invalid risk normalization checkpoint')
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite risk checkpoint')
    model.eval().requires_grad_(False);model.to(device)
    return model,threshold
