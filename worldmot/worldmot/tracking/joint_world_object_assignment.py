"""History-centred, bounded two-camera joint set-packing optimization.

Each group is a historical ID with NULL / one-view / two-view alternatives,
or a high-confidence birth pair. Singleton births are the unassigned baseline.
Small conflict components are solved exactly; larger ones use a deterministic
beam. No camera pair is committed before the historical alternatives compete.
Only NumPy is required. Search caps are part of the method manifest.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .world_object_config import WorldObjectConfig


@dataclass(frozen=True)
class AssignmentOption:
    global_id: int | None
    indices: tuple[int, ...]
    benefit: float
    reason: str


@dataclass
class JointAssignmentResult:
    selected: list[AssignmentOption]
    unmatched: list[int]
    diagnostics: dict = field(default_factory=dict)


def _similarity(a, b):
    return float(np.clip(np.dot(a, b), -1, 1))


def spatial_distance(center_a, covariance_a, center_b, covariance_b):
    delta = np.asarray(center_a) - np.asarray(center_b)
    cov = np.asarray(covariance_a) + np.asarray(covariance_b) + np.eye(3) * 0.01
    return float(np.sqrt(max(0, delta @ np.linalg.pinv(cov) @ delta)))


def spatial_distance_matrix(left,right):
    def values(items):
        centers=[];covariances=[]
        for x in items:
            valid=x.geometry_valid if hasattr(x,"center") else x.cross_view_fusion_eligible
            centers.append((x.center if hasattr(x,"center") else x.world_center) if valid else np.zeros(3))
            covariances.append((x.covariance if hasattr(x,"covariance") else x.world_covariance) if valid else np.eye(3)*1e6)
        return np.asarray(centers),np.asarray(covariances)
    a,ca=values(left);b,cb=values(right)
    delta=a[:,None]-b[None,:];cov=ca[:,None]+cb[None,:]+np.eye(3)*.01
    solved=np.linalg.solve(cov,delta[...,None])[...,0]
    return np.sqrt(np.maximum(0,(delta*solved).sum(-1)))


class JointWorldObjectAssignment:
    def __init__(self, config: WorldObjectConfig):
        self.config = config
        self.last_birth_candidate_edges=set()

    def _history_evidence(self, q, o,app_override=None,spatial_override=None,learned_margin=None):
        cfg = self.config
        app = None
        if cfg.use_appearance and (q.identity_embedding is None or not o.feature_valid):
            return None
        if cfg.use_appearance and q.identity_embedding is not None and o.feature_valid:
            app = _similarity(q.identity_embedding, o.roi_feature) if app_override is None else float(app_override)
            if app < cfg.min_appearance_similarity and (learned_margin is None or learned_margin<cfg.learned_identity_recovery_logit):
                return None
        spatial = None
        if cfg.use_geometry and q.geometry_valid and o.cross_view_fusion_eligible:
            spatial = spatial_distance(q.center, q.covariance, o.world_center, o.world_covariance) if spatial_override is None else float(spatial_override)
            if spatial > cfg.geometry_gate:
                # A stale position must not forbid a strong reappearance.
                if not (q.missed > 0 and app is not None and app >= cfg.recovery_similarity):
                    return None
                return 0.6 * app, "VISUAL_RECOVERY_SPATIAL_CONFLICT"
        if app is None and spatial is None:
            return None
        evidence = ((app if app is not None else 0.0)
                    + (0.5 * np.exp(-0.5 * (spatial / 3) ** 2) if spatial is not None else 0.0))
        return float(evidence), "HISTORY_EXPLAINS_CURRENT_OBSERVATION"

    def _pair_evidence(self, a, b, birth=False, learned_probability=None):
        cfg = self.config
        if a.camera_id == b.camera_id:
            return None
        app = None
        if cfg.use_appearance:
            if not (a.feature_valid and b.feature_valid):
                return None
            indices=(self._current_index.get(id(a)),self._current_index.get(id(b))) if hasattr(self,"_current_index") else (None,None)
            app = (float(self._current_appearance[indices]) if None not in indices and self._current_appearance is not None
                   else _similarity(a.roi_feature,b.roi_feature))
            # A learned interaction can link dissimilar views; cosine is only
            # the explicit heuristic baseline, not a hard learned-birth gate.
            if learned_probability is None and app < (cfg.joint_birth_similarity if birth else cfg.min_appearance_similarity):
                return None
        geom = None
        if cfg.use_geometry and a.cross_view_fusion_eligible and b.cross_view_fusion_eligible:
            indices=(self._current_index.get(id(a)),self._current_index.get(id(b))) if hasattr(self,"_current_index") else (None,None)
            geom = (float(self._current_spatial[indices]) if None not in indices and self._current_spatial is not None
                    else spatial_distance(a.world_center,a.world_covariance,b.world_center,b.world_covariance))
            if geom > cfg.geometry_gate:
                return None
        # New objects require both cues in the full method; no forced fusion
        # from centre proximity alone. Visual-only methods have a stricter ROI gate.
        if birth and cfg.use_geometry and geom is None and learned_probability is None and (app is None or app<0.97):
            return None
        if learned_probability is not None:
            if learned_probability<cfg.birth_pair_threshold:return None
            return float(learned_probability)
        if app is None and geom is None:
            return None
        return float((app or 0.0) + (0.3 * np.exp(-geom / 3) if geom is not None else 0.0))

    def assign(self, history, observations, learned=None):
        cfg = self.config
        n, m = len(observations), len(history)
        self.last_birth_candidate_edges=set()
        if n > cfg.max_observations or m > cfg.max_history:
            return JointAssignmentResult([], list(range(n)), {
                "fallback": "SEARCH_LIMIT_SINGLETON_TENTATIVE_BIRTHS", "observations": n,
                "history": m, "expanded_states": 0, "constraints_satisfied": True})
        groups = []
        comparisons = 0
        self._current_index={id(o):j for j,o in enumerate(observations)}
        of=np.array([o.roi_feature if o.feature_valid else np.zeros(cfg.feature_dim) for o in observations],np.float32).reshape(n,cfg.feature_dim)
        hf=np.array([q.identity_embedding if q.identity_embedding is not None else np.zeros(cfg.feature_dim) for q in history],np.float32).reshape(m,cfg.feature_dim)
        app_matrix=np.clip(hf@of.T,-1,1) if cfg.use_appearance else None
        self._current_appearance=np.clip(of@of.T,-1,1) if cfg.use_appearance else None
        space_matrix=spatial_distance_matrix(history,observations) if cfg.use_geometry and m and n else None
        self._current_spatial=spatial_distance_matrix(observations,observations) if cfg.use_geometry and n else None
        for qi, q in enumerate(history):
            per_view = {}
            for j, o in enumerate(observations):
                comparisons += 1
                margin=None if learned is None else float(learned["assignment_logits"][qi,j]-learned["null_logits"][qi])
                evidence = self._history_evidence(q,o,None if app_matrix is None else app_matrix[qi,j],
                                                   None if space_matrix is None else space_matrix[qi,j],margin)
                if evidence is None:
                    continue
                value, reason = evidence
                if learned is not None:
                    # Candidate-ID-specific logits, relative to the object's NULL.
                    value = (1-cfg.learned_weight)*value + cfg.learned_weight * float(np.tanh(
                        learned["assignment_logits"][qi, j] - learned["null_logits"][qi]))
                    value += 0.1 * float(np.tanh(learned["existence_logits"][qi]))
                per_view.setdefault(o.camera_id, []).append((value, j, reason))
            options = []
            for view in per_view:
                per_view[view] = sorted(per_view[view], key=lambda x: (-x[0], observations[x[1]].key))[:cfg.top_k]
                options.extend(AssignmentOption(q.global_id, (j,), value, reason)
                               for value, j, reason in per_view[view] if value > 0)
            views = sorted(per_view)
            if len(views) == 2:
                for va, a, ra in per_view[views[0]]:
                    for vb, b, rb in per_view[views[1]]:
                        pair = self._pair_evidence(observations[a], observations[b])
                        if pair is not None:
                            options.append(AssignmentOption(q.global_id, tuple(sorted((a, b))),
                                                            va + vb + 0.2 * pair,
                                                            "VISUAL_RECOVERY_SPATIAL_CONFLICT" if "SPATIAL_CONFLICT" in ra+rb else "JOINT_HISTORY_TWO_VIEW"))
            if options:
                groups.append(sorted(options, key=lambda x: (-x.benefit, x.indices)))
        # Joint births compete in the same conflict graph as historical IDs.
        views = sorted({o.camera_id for o in observations})
        birth_pairs = {}
        if len(views) == 2:
            for a, oa in enumerate(observations):
                if oa.camera_id != views[0]:
                    continue
                options = []
                for b, ob in enumerate(observations):
                    if ob.camera_id != views[1]:
                        continue
                    comparisons += 1
                    probability=None
                    if learned is not None and cfg.use_learned_joint_birth and "birth_pair_logits" in learned:
                        probability=_sigmoid(learned["birth_pair_logits"][a,b])
                    pair = self._pair_evidence(oa, ob, birth=True,learned_probability=probability)
                    if pair is None:
                        continue
                    birth_prob = 1.0
                    if learned is not None:
                        birth_prob = min(_sigmoid(learned["birth_logits"][a]), _sigmoid(learned["birth_logits"][b]))
                    if birth_prob >= cfg.birth_threshold:
                        if probability is None:
                            value=0.25*pair*birth_prob;reason="HEURISTIC_JOINT_BIRTH"
                        else:
                            threshold_logit=np.log(cfg.birth_pair_threshold/(1-cfg.birth_pair_threshold))
                            value=cfg.birth_pair_weight*max(0.,float(learned["birth_pair_logits"][a,b])-threshold_logit)*birth_prob
                            reason="LEARNED_CROSS_VIEW_JOINT_BIRTH"
                        options.append(AssignmentOption(None,(a,b),value,reason))
                for option in sorted(options, key=lambda x: (-x.benefit, x.indices))[:cfg.top_k]:
                    birth_pairs[option.indices] = option
                    self.last_birth_candidate_edges.add(option.indices)
            groups.extend([[x] for _, x in sorted(birth_pairs.items())])
        # Groups sharing an observation form an independent conflict component.
        parent = list(range(len(groups)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        owners = {}
        for gi, group in enumerate(groups):
            for j in {j for option in group for j in option.indices}:
                if j in owners:
                    parent[root(gi)] = root(owners[j])
                owners[j] = gi
        components = {}
        for gi in range(len(groups)):
            components.setdefault(root(gi), []).append(gi)
        selected, audits = [], []
        for component in components.values():
            selected_part, audit = self._solve_component([groups[i] for i in component])
            selected.extend(selected_part)
            audits.append(audit)
        used = [j for option in selected for j in option.indices]
        if len(used) != len(set(used)):
            raise AssertionError("joint assignment violated one-detection/one-ID")
        gids = [o.global_id for o in selected if o.global_id is not None]
        if len(gids) != len(set(gids)):
            raise AssertionError("joint assignment repeated historical ID")
        return JointAssignmentResult(selected, [j for j in range(n) if j not in set(used)], {
            "solver": "COMPONENT_EXACT_OR_BOUNDED_BEAM", "components": audits,
            "top_k": cfg.top_k, "beam_width": cfg.beam_width,
            "candidate_comparisons": comparisons,
            "expanded_states": sum(x["expanded_states"] for x in audits),
            "learned_birth_pair_used":bool(learned is not None and cfg.use_learned_joint_birth and "birth_pair_logits" in learned),
            "constraints_satisfied": True, "premerged_pairs": False})

    def _solve_component(self, groups):
        budget = self.config.max_exact_states
        combinations = math.prod(len(g) + 1 for g in groups)
        exact = combinations <= budget
        width = budget if exact else self.config.beam_width
        states = [(0.0, frozenset(), ())]
        expanded = 0
        for group in groups:
            next_states = []
            for value, used, chosen in states:
                next_states.append((value, used, chosen))  # NULL alternative
                for option in group:
                    expanded += 1
                    if not used.intersection(option.indices):
                        next_states.append((value + option.benefit, used.union(option.indices), chosen + (option,)))
            next_states.sort(key=lambda x: (-x[0], tuple((o.global_id or 0, o.indices) for o in x[2])))
            states = next_states[:width]
        return list(states[0][2]), {"mode": "EXACT" if exact else "BEAM", "groups": len(groups),
                                  "state_width_limit": width, "expanded_states": expanded,
                                  "enumeration_upper_bound": combinations if exact else None}


def _sigmoid(value):
    return float(1 / (1 + np.exp(-np.clip(value, -60, 60))))
