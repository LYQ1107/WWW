from torch import nn
from torch.nn import functional as F
from .visual_state_encoder import ReaderBlock

class TypedQuestionReader(nn.Module):
    def __init__(self,d=128,heads=4):
        super().__init__()
        self.visual_projection = nn.Linear(1152,d)
        self.context_projection = nn.Sequential(nn.Linear(64,d),nn.GELU(),nn.Linear(d,d))
        self.type_descriptor = nn.Embedding(3,d)
        self.reader = ReaderBlock(d,heads)
        self.output_norm = nn.LayerNorm(d)

    def forward(self,memory,mask,questions,context,mode='dynamic'):
        descriptor = self.type_descriptor(questions.types)
        dynamic = self.visual_projection(F.normalize(questions.visual,dim=-1,eps=1e-8)) + self.context_projection(context)
        initial = descriptor if mode == 'fixed' else descriptor + dynamic
        result = initial if mode == 'no_reader' else self.reader(initial,memory,mask)
        return self.output_norm(result)*questions.mask[...,None]
