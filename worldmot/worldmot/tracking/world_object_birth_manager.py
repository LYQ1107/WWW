"""Detection-supported births after historical assignment, without neural heads."""
from dataclasses import dataclass,field
import numpy as np
from .joint_world_object_assignment import AssignmentOption
from ..training.world_object_identity_targets import box_iou
from ..models.world_object_v5_geometry import spatial_pair_evidence


@dataclass
class BirthPlan:
    selected: list = field(default_factory=list)
    duplicates: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


class WorldObjectBirthManager:
    def __init__(self,config):self.config=config

    def _duplicate(self,a,b,already_assigned,indices,history_hint):
        cfg=self.config
        if a.camera_id!=b.camera_id:return False
        ia,ib=indices
        if ia in already_assigned and ib in already_assigned and already_assigned[ia]!=already_assigned[ib]:return False
        # Distinct strong historical explanations protect adjacent real people.
        if history_hint.get(ia) is not None and history_hint.get(ib) is not None and history_hint[ia]!=history_hint[ib]:return False
        cosine=float(a.roi_feature@b.roi_feature)
        if cosine<cfg.duplicate_min_roi_cosine or box_iou(a.bbox_xywh,b.bbox_xywh)<cfg.duplicate_min_iou:return False
        lo=np.maximum(a.bbox_xywh[:2],b.bbox_xywh[:2]);hi=np.minimum(a.bbox_xywh[:2]+a.bbox_xywh[2:],b.bbox_xywh[:2]+b.bbox_xywh[2:])
        overlap=np.maximum(hi-lo,0).prod();containment=overlap/max(1e-12,min(a.bbox_xywh[2:].prod(),b.bbox_xywh[2:].prod()))
        if containment<cfg.duplicate_min_containment:return False
        exact=bool(np.allclose(a.bbox_xywh,b.bbox_xywh,rtol=0,atol=1e-6))
        same_current_history=history_hint.get(ia) is not None and history_hint.get(ia)==history_hint.get(ib)
        if exact:return True
        if not same_current_history:return False
        if a.geometry_valid and b.geometry_valid:
            c=a.world_covariance+b.world_covariance;floor=max(float(np.trace(c))/3*1e-4,1e-10)
            delta=a.world_center-b.world_center
            try:return bool(delta@np.linalg.solve(c+np.eye(3)*floor,delta)<.25)
            except np.linalg.LinAlgError:return False
        return False  # Occluded overlap and missing depth remain ambiguous.

    def form(self,observations,unmatched,assigned,context,*,history_hint=None):
        cfg=self.config;history_hint=history_hint or {};plan=BirthPlan()
        remaining=set(map(int,unmatched));anchor_order=sorted(range(len(observations)),key=lambda j:(j not in assigned,-observations[j].score,observations[j].key))
        for j in sorted(list(remaining),key=lambda j:(-observations[j].score,observations[j].key)):
            if j not in remaining:continue
            for k in anchor_order:
                if k==j or k in plan.duplicates:continue
                if k not in assigned and (observations[k].score,tuple(-x for x in observations[k].key))<(observations[j].score,tuple(-x for x in observations[j].key)):continue
                if self._duplicate(observations[j],observations[k],assigned,(j,k),history_hint):
                    plan.duplicates[j]=k;plan.reasons[j]="TRACK_AWARE_REDUNDANT_CURRENT_DETECTION"
                    remaining.remove(j);break
        edges=[];pair_records=[]
        ordered=sorted(remaining,key=lambda j:observations[j].key)
        for at,a in enumerate(ordered):
            oa=observations[a]
            for b in ordered[at+1:]:
                ob=observations[b]
                if oa.camera_id==ob.camera_id:continue
                app=float(oa.roi_feature@ob.roi_feature)
                if app<min(cfg.birth_geometry_min_roi_cosine,cfg.birth_visual_only_min_roi_cosine):continue
                geometric=spatial_pair_evidence(oa,ob,context,cfg)
                reason=None;score=0.
                if (geometric["reliable"] and app>=cfg.birth_geometry_min_roi_cosine
                        and geometric["mahalanobis"]<=cfg.geometry_pair_max_mahalanobis and geometric["support_gap"]<=2.5):
                    projected=(not geometric["reprojection_available"] or min(geometric["coverage_a_to_b"],geometric["coverage_b_to_a"])>=cfg.birth_min_reprojection_coverage)
                    if projected:
                        score=app+.3*np.exp(-geometric["mahalanobis"]**2/12)-.1*geometric["support_gap"]
                        reason="NEW_WORLD_OBJECT_SUPPORT_AND_APPEARANCE"
                elif not geometric["reliable"] and app>=cfg.birth_visual_only_min_roi_cosine:
                    score=app;reason="NEW_WORLD_OBJECT_STRONG_VISUAL_ABSTAINED_GEOMETRY"
                if reason is not None:edges.append((float(score),a,b,reason))
                if app>=cfg.birth_geometry_min_roi_cosine:
                    pair_records.append({"a":a,"b":b,"ROI_cosine":app,"geometry":geometric,"admissible":reason is not None})
        accepted=[]
        for score,a,b,reason in sorted(edges,key=lambda e:(-e[0],observations[e[1]].key,observations[e[2]].key)):
            if a not in remaining or b not in remaining:continue
            alternatives_a=[s for s,x,y,_ in edges if (x==a or y==a) and {x,y}!={a,b}]
            alternatives_b=[s for s,x,y,_ in edges if (x==b or y==b) and {x,y}!={a,b}]
            margin=min(score-max(alternatives_a,default=score-1),score-max(alternatives_b,default=score-1))
            if margin<cfg.birth_pair_competition_margin:continue
            indices=tuple(sorted((a,b)));plan.selected.append(AssignmentOption(None,indices,score,reason))
            remaining.difference_update(indices);accepted.append(indices)
            for j in indices:plan.reasons[j]=reason
        for j in sorted(remaining,key=lambda j:observations[j].key):
            o=observations[j];reason="HIGH_QUALITY_SINGLETON_BIRTH" if o.score>=cfg.birth_high_score else "BOUNDARY_SCORE_TENTATIVE_SINGLETON"
            plan.selected.append(AssignmentOption(None,(j,),0.,reason));plan.reasons[j]=reason
        plan.diagnostics={"births":len(plan.selected),"joint_births":len(accepted),"duplicate_suppressed":len(plan.duplicates),
                          "birth_pair_edges":len(edges),"birth_pair_audit":pair_records,
                          "all_unmatched_have_decision":set(unmatched)==set(plan.reasons)}
        if not plan.diagnostics["all_unmatched_have_decision"]:raise AssertionError("UNEXPLAINED_BIRTH_CANDIDATE")
        return plan
