"""Bounded historical set packing. No birth candidate enters this optimizer."""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from .joint_world_object_assignment import AssignmentOption,JointAssignmentResult,JointWorldObjectAssignment
from ..models.world_object_v5_geometry import spatial_pair_evidence


class HistoricalWorldObjectAssignment:
    def __init__(self,config):
        self.config=config
        self.solver=JointWorldObjectAssignment(config)  # only its unchanged bounded component solver

    def _solve(self,groups):
        parent=list(range(len(groups)));owners={}
        def root(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,options in enumerate(groups):
            for j in {j for o in options for j in o.indices}:
                if j in owners:parent[root(i)]=root(owners[j])
                owners[j]=i
        parts=defaultdict(list)
        for i,g in enumerate(groups):parts[root(i)].append(g)
        chosen=[];audit=[]
        estimated=sum(sum(len(g)+1 for g in groups)*self.config.beam_width for groups in parts.values())
        if estimated>self.config.solver_expansion_budget:return None,{"reason":"SOLVER_EXPANSION_BUDGET","estimate":estimated}
        for part in parts.values():
            selected,record=self.solver._solve_component(part);chosen.extend(selected);audit.append(record)
        return chosen,{"components":audit,"estimated_expansions":estimated,"expanded_states":sum(x["expanded_states"] for x in audit)}

    def _compatible(self,a,b,observations,context):
        if not self.config.use_geometry:return True
        geo=spatial_pair_evidence(observations[a],observations[b],context,self.config)
        # This rejects a contradictory TWO-view grouping only when both actual
        # supports are reliable; each historical singleton remains admissible.
        if geo["reliable"]:
            if geo["mahalanobis"]>self.config.geometry_pair_max_mahalanobis or geo["support_gap"]>3:return False
            if geo["reprojection_available"] and min(geo["coverage_a_to_b"],geo["coverage_b_to_a"])<.05:return False
        return True

    def fallback(self,history,observations,context):
        """Keep high-confidence historical explanations even when search is capped."""
        if not history or not observations:return JointAssignmentResult([],list(range(len(observations))),{"solver":"EMPTY_HISTORY_OR_OBSERVATIONS"})
        features=np.asarray([q.identity_embedding if q.identity_embedding is not None else np.zeros(self.config.feature_dim) for q in history])
        app=features@np.asarray([o.roi_feature for o in observations]).T
        candidates=[]
        for j,o in enumerate(observations):
            ranked=np.argsort(-app[:,j],kind="stable");i=int(ranked[0]);value=app[i,j]
            rival=float(app[ranked[1],j]) if len(ranked)>1 else -1.
            same_camera=[k for k,x in enumerate(observations) if x.camera_id==o.camera_id and k!=j]
            other=max((app[i,k] for k in same_camera),default=-1.)
            if value>=.9 and value-max(rival,other)>=.05:candidates.append((float(value),i,j))
        selected=defaultdict(list);used=set();slots=set()
        for value,i,j in sorted(candidates,key=lambda x:(-x[0],history[x[1]].global_id,observations[x[2]].key)):
            slot=i,observations[j].camera_id
            if j in used or slot in slots:continue
            if any(not self._compatible(j,k,observations,context) for k in selected[i]):continue
            selected[i].append(j);used.add(j);slots.add(slot)
        options=[AssignmentOption(history[i].global_id,tuple(sorted(js)),0.,"CONSERVATIVE_HISTORICAL_BUDGET_FALLBACK") for i,js in sorted(selected.items())]
        return JointAssignmentResult(options,[j for j in range(len(observations)) if j not in used],
            {"solver":"CONSERVATIVE_HISTORICAL_FALLBACK","historical_observations_retained":len(used),"constraints_satisfied":True,"birth_candidates_in_optimizer":0})

    def assign(self,history,observations,margins,context,geometry=None):
        cfg=self.config;m,n=len(history),len(observations)
        if margins is None or m*n>262144:return self.fallback(history,observations,context)
        if margins.shape!=(m,n) or not np.isfinite(margins).all():raise ValueError("INVALID_MEMBERSHIP_MARGINS")
        values=margins.astype(float).copy()
        if geometry is not None and cfg.rich_geometry and m and n:
            weight=geometry[:,:,6];z=geometry[:,:,:3]/np.maximum(weight[:,:,None],1e-12)
            values+=cfg.geometry_bonus_bound*weight*(2*np.exp(-np.square(z).sum(-1)/6)-1)
        selected={};used=set();details=[];slots=set()
        # ByteTrack-inspired score precedence; low-score recovery cannot replace
        # a high-score historical observation already committed this timestamp.
        for high in (True,False):
            groups=[]
            for i,q in enumerate(history):
                per=defaultdict(list)
                for j,o in enumerate(observations):
                    if j in used or (i,o.camera_id) in slots or (o.score>=cfg.birth_high_score)!=high:continue
                    threshold=cfg.membership_min_margin if high else cfg.low_score_recovery_min_margin
                    if values[i,j]<=threshold:continue
                    if any(not self._compatible(j,k,observations,context) for k in selected.get(i,[])):continue
                    per[o.camera_id].append(j)
                options=[]
                for camera,js in per.items():
                    per[camera]=sorted(js,key=lambda j:(-values[i,j],observations[j].key))[:cfg.top_k]
                    options.extend(AssignmentOption(q.global_id,(j,),float(values[i,j]),"HISTORICAL_MEMBERSHIP_HIGH" if high else "HISTORICAL_LOW_SCORE_RECOVERY") for j in per[camera])
                views=sorted(per)
                if len(views)==2:
                    for a in per[views[0]]:
                        for b in per[views[1]]:
                            if self._compatible(a,b,observations,context):
                                options.append(AssignmentOption(q.global_id,tuple(sorted((a,b))),float(values[i,a]+values[i,b]),"HISTORICAL_TWO_CAMERA_MEMBERSHIP"))
                if options:groups.append(sorted(options,key=lambda x:(-x.benefit,x.indices)))
            part,record=self._solve(groups)
            if part is None:return self.fallback(history,observations,context)
            details.append(record)
            indices={q.global_id:i for i,q in enumerate(history)}
            for option in part:
                i=indices[option.global_id];selected.setdefault(i,[]).extend(option.indices)
                for j in option.indices:used.add(j);slots.add((i,observations[j].camera_id))
        options=[AssignmentOption(history[i].global_id,tuple(sorted(js)),float(sum(values[i,j] for j in js)),
                   "HISTORICAL_TWO_CAMERA_MEMBERSHIP" if len(js)>1 else "HISTORICAL_MEMBERSHIP") for i,js in sorted(selected.items())]
        if any(len({observations[j].camera_id for j in o.indices})!=len(o.indices) or o.global_id is None for o in options):
            raise AssertionError("HISTORICAL_SOLVER_CONSTRAINT_VIOLATION")
        return JointAssignmentResult(options,[j for j in range(n) if j not in used],
            {"solver":"HISTORY_ONLY_COMPONENT_EXACT_OR_BEAM","stages":details,"constraints_satisfied":True,
             "birth_candidates_in_optimizer":0,"unmatched_are_birth_manager_inputs":True})
