import torch
from torch import nn
from torch.nn import functional as F

class ReaderBlock(nn.Module):
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.query_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d,heads,dropout=0.,batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))

    def forward(self, query, memory, mask):
        if not query.shape[1]:
            return query
        m = self.memory_norm(memory)
        read = self.attention(self.query_norm(query),m,m,key_padding_mask=~mask,need_weights=False)[0]
        result = query + read
        return result + self.ffn(self.ffn_norm(result))

class StateEncoder(nn.Module):
    def __init__(self,d=128,heads=4):
        super().__init__()
        self.visual_projection = nn.Linear(1152,d)
        self.metadata_projection = nn.Linear(8,d)
        self.latents = nn.Parameter(torch.randn(1,8,d)*.02)
        self.pool = ReaderBlock(d,heads)
        self.fusion = ReaderBlock(d,heads)

    def forward(self,state):
        values = self.visual_projection(F.normalize(state.visual,dim=-1,eps=1e-8)) + self.metadata_projection(state.metadata)
        memory = self.pool(self.latents.expand(len(values),-1,-1),values,state.mask)
        mask = torch.ones(memory.shape[:2],dtype=torch.bool,device=memory.device)
        return self.fusion(memory,memory,mask),mask
