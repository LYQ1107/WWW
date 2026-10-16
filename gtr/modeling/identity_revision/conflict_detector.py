"""Conservative, interpretable precommit conflict evidence; no GT attributes."""
from dataclasses import dataclass,asdict
import math
from .types import ConflictState


@dataclass(frozen=True)
class ConflictEvidence:
    selected_id: int
    local_id: int
    local_candidate_legal: bool
    local_consecutive_frames: int
    local_iou: float
    local_iou_margin: float
    local_cosine: float
    selected_history_cosine: float
    normalized_assignment_margin: float


@dataclass(frozen=True)
class ConflictThresholds:
    minimum_local_frames: int=3
    minimum_local_iou: float=.7
    minimum_iou_margin: float=.3
    minimum_local_cosine: float=.9
    maximum_selected_cosine: float=.7
    maximum_assignment_margin: float=.02


class FixedConflictDetector:
    def __init__(self,thresholds=ConflictThresholds()):self.thresholds=thresholds

    def classify(self,evidence):
        if not isinstance(evidence,ConflictEvidence):raise TypeError('Only whitelisted GT-free conflict evidence')
        e=evidence;t=self.thresholds
        numeric=[e.local_iou,e.local_iou_margin,e.local_cosine,e.selected_history_cosine,e.normalized_assignment_margin]
        if not all(math.isfinite(v) for v in numeric):return ConflictState.UNKNOWN,('nonfinite_or_missing_evidence',)
        if e.selected_id<1 or e.local_id<1 or not e.local_candidate_legal:
            return ConflictState.UNKNOWN,('no_legal_old_identity_or_local_candidate',)
        stable=e.local_consecutive_frames>=t.minimum_local_frames
        geometry=e.local_iou>=t.minimum_local_iou and e.local_iou_margin>=t.minimum_iou_margin
        visual=e.local_cosine>=t.minimum_local_cosine
        if stable and geometry and visual and e.local_id==e.selected_id:
            return ConflictState.SAFE,('stable_local_supports_current_proposal',)
        mismatch=e.local_id!=e.selected_id
        history_conflict=e.selected_history_cosine<t.maximum_selected_cosine
        weak=e.normalized_assignment_margin<t.maximum_assignment_margin
        if stable and geometry and visual and mismatch and history_conflict and weak:
            return ConflictState.SUSPECT,('three_frame_local_chain','unique_local_overlap','strong_local_visual_support',
                'contradicting_selected_history','weak_native_joint_choice')
        return ConflictState.UNKNOWN,('insufficient_joint_conflict_evidence',)

    def protocol(self):
        return dict(type='fixed_conjunctive_precommit_conflict_v1',thresholds=asdict(self.thresholds),
            output='SAFE/SUSPECT/UNKNOWN; SUSPECT is a testable risk hypothesis, not certified GT',
            same_camera_overlap_alone_sufficient=False,cross_camera_appearance_alone_sufficient=False,
            simultaneous_cross_camera_observation_is_conflict=False)
