"""Deployment-only relation records. No GT/owner supervision fields."""
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, order=True)
class SourceKey:
    video: int
    frame: int
    camera: int
    image: int
    row: int

    def __post_init__(self):
        if min(self.video,self.frame,self.camera,self.image)<1 or self.row<0:
            raise ValueError('Invalid physical source key')


class ConflictState(str,Enum):
    SAFE='SAFE'
    SUSPECT='SUSPECT'
    UNKNOWN='UNKNOWN'


@dataclass
class IdentityFragment:
    fragment_key: SourceKey
    original_global_id: int
    current_global_id: int
    camera: int
    start_frame: int
    latest_frame: int
    observation_source_keys: list = field(default_factory=list)
    observations_seen: int = 0
    relation_version: int = 0
    published_through: int = 0
    state: ConflictState = ConflictState.UNKNOWN


@dataclass(frozen=True)
class RevisionAction:
    action_type: str
    source: SourceKey
    target_global_id: int
    original_native_id: int
    evidence_keys: tuple
    causal_timestamp: int

    def __post_init__(self):
        if self.action_type not in ('KEEP','VETO','PROVISIONAL','SPLIT','RELINK'):
            raise ValueError('Unknown revision action')
        if self.causal_timestamp!=self.source.frame or any(k.video!=self.source.video or k.frame>self.causal_timestamp for k in self.evidence_keys):
            raise ValueError('Noncausal identity action')
