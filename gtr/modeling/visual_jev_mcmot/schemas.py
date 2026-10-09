"""Runtime evidence only. Supervision and identity integers are not NN features."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple
import torch

class QuestionType(IntEnum):
    MATCH_ID = 0
    REACTIVATE_ID = 1
    UPDATE_MEMORY = 2

class ActionType(IntEnum):
    ASSOCIATE_EXISTING = 0
    DEFER_TO_REACTIVATION = 1
    REACTIVATE = 2
    START_NEW = 3
    WRITE_MEMORY = 4
    KEEP_MEMORY = 5
    ABSTAIN_OR_FALLBACK = 6

@dataclass(frozen=True)
class ActionOption:
    question_type: QuestionType
    semantic_action_type: ActionType
    detection_reference: Tuple[int, int, int, int]
    identity_reference: Optional[int] = None
    legal: bool = True
    native_execution_payload: Optional[dict] = None
    evidence_features: Optional[torch.Tensor] = None
    visual_identity_tokens: Optional[torch.Tensor] = None
    temporal_context: Optional[dict] = None

@dataclass(frozen=True)
class OnlineVisualState:
    visual: torch.Tensor
    metadata: torch.Tensor
    mask: torch.Tensor

@dataclass(frozen=True)
class QuestionDescriptor:
    visual: torch.Tensor
    context: torch.Tensor
    types: torch.Tensor
    mask: torch.Tensor

@dataclass(frozen=True)
class OptionTensors:
    visual: torch.Tensor
    visual_mask: torch.Tensor
    evidence: torch.Tensor
    kinds: torch.Tensor
    mask: torch.Tensor

def validate(state, questions, options):
    b,t,v = state.visual.shape
    if v != 1152 or state.metadata.shape != (b,t,8) or state.mask.shape != (b,t):
        raise ValueError('invalid visual state shape')
    if t == 0 or not state.mask.any(-1).all():
        raise ValueError('visual state needs a genuine or empty-state sentinel token')
    bq,q,vq = questions.visual.shape
    if bq != b or vq != 1152 or questions.context.shape != (b,q,64) or questions.types.shape != (b,q) or questions.mask.shape != (b,q):
        raise ValueError('invalid question shape')
    bo,qo,k,h,vo = options.visual.shape
    if (bo,qo,h,vo) != (b,q,3,1152) or options.visual_mask.shape != (b,q,k,3) or options.evidence.shape != (b,q,k,12) or options.kinds.shape != (b,q,k) or options.mask.shape != (b,q,k):
        raise ValueError('invalid option shape')
    for x in (state.visual,state.metadata,questions.visual,questions.context,options.visual,options.evidence):
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite runtime evidence')
    if ((questions.types < 0) | (questions.types > 2)).any() or ((options.kinds < 0) | (options.kinds > 5)).any():
        raise ValueError('invalid typed action')
    allowed = ((questions.types[...,None] == 0) & (options.kinds <= 1)) | ((questions.types[...,None] == 1) & ((options.kinds == 2) | (options.kinds == 3))) | ((questions.types[...,None] == 2) & (options.kinds >= 4))
    if (options.mask & ~allowed).any():
        raise ValueError('action is illegal for this question type')
    if (options.mask & ~questions.mask[...,None]).any():
        raise ValueError('padded question cannot own legal actions')
    memory=(questions.types==2)&questions.mask
    if memory.any() and (((options.mask&(options.kinds==4)).sum(-1)[memory]!=1).any() or ((options.mask&(options.kinds==5)).sum(-1)[memory]!=1).any()):
        raise ValueError('MEMORY has exactly one WRITE and one KEEP option')
    if (options.mask & ~options.visual_mask.any(-1)).any():
        raise ValueError('legal option must have genuine visual evidence')

def masked_softmax(logits, mask):
    # Empty legal sets return exactly zero; their executor receives fallback.
    maximum = logits.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True) if logits.shape[-1] else logits.sum(-1,keepdim=True)
    maximum = torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    values = torch.exp((logits-maximum).masked_fill(~mask,-torch.inf))
    return values / values.sum(-1,keepdim=True).clamp_min(1e-12)
