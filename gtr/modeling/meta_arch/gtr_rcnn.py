import cv2
import os
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference
from ..jev_runtime import (
    DecisionTraceWriter,
    JEVRuntimePolicy,
    build_controller_from_checkpoint,
)
from ..jev_state import (
    build_state_values,
    encode_state,
    legacy_acceptance_threshold,
)
from ..jev_assignment import constrained_hungarian
from ..jev_perception_cache import FrozenPerceptionCache, FrozenPerceptionCacheWriter
from tqdm import tqdm
import time
import copy
from detectron2.structures import ImageList, Instances


class poss_ids:
    poss_ids = set()

    def __init__(self):
        pass

class old_ids:
    old_ids = set()
    def __init__(self):
        pass
    
class old_reids:
    old_reids = []
    def __init__(self):
        pass

@META_ARCH_REGISTRY.register()
class GTRRCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        self.with_attention = kwargs.pop('with_attention')
        self.thred_bank = kwargs.pop('thred')
        self.bank_size = kwargs.pop('bank_size')
        self.with_bank = kwargs.pop('with_bank')
        self.multi_modal = kwargs.pop('multi_modal')
        self.jev_enabled = bool(kwargs.pop('jev_enabled'))
        self.jev_mode = str(kwargs.pop('jev_mode')).lower()
        self.jev_state_dim = int(kwargs.pop('jev_state_dim'))
        self.jev_max_reassociate = int(kwargs.pop('jev_max_reassociate'))
        self.jev_trace_path = str(kwargs.pop('jev_trace_path'))
        self.jev_controller_weights = str(kwargs.pop('jev_controller_weights'))
        candidate_enabled = bool(kwargs.pop('jev_candidate_enabled', False))
        candidate_name = str(kwargs.pop('jev_candidate_policy', 'gmt_compat'))
        candidate_weights = str(kwargs.pop('jev_candidate_scripted_model', ''))
        if candidate_enabled and (not self.jev_enabled or self.jev_mode != 'off'):
            raise ValueError('candidate values require MODEL.JEV.ENABLED=true and MODE=off for frozen lifecycle control')
        super().__init__(**kwargs)
        self.jev_candidate_policy = None
        self.jev_candidate_commit_observer = None
        self._jev_candidate_new_rows = ()
        if candidate_enabled:
            from ..jev_candidate_policy import CandidateValuePolicy
            candidate_model = None
            if candidate_name == 'model':
                if not candidate_weights:
                    raise ValueError('candidate model mode needs an explicit scripted model')
                candidate_model = torch.jit.load(candidate_weights, map_location=self.device).eval()
            self.jev_candidate_policy = CandidateValuePolicy(candidate_name, candidate_model)
        self.jev_policy = None
        self.jev_trace_writer = None
        self._jev_context = {}
        self._jev_trajectory_rng = None
        cache_path = os.environ.get('JEV_PERCEPTION_CACHE_PATH', '').strip()
        self.jev_perception_cache = (
            FrozenPerceptionCacheWriter(cache_path) if cache_path else None
        )
        # A native GMT trace can be replayed against the exact immutable
        # detector/ReID source used by the mutable formal runner.  This is a
        # read-only diagnostic path and is deliberately separate from the
        # writer above so a trace run cannot overwrite canonical cache files.
        cache_read_path = os.environ.get('JEV_PERCEPTION_CACHE_READ_PATH', '').strip()
        self.jev_perception_cache_reader = (
            FrozenPerceptionCache(cache_read_path) if cache_read_path else None
        )
        if self.jev_enabled:
            if self.jev_mode != 'off' and not self.jev_controller_weights:
                raise ValueError(
                    'MODEL.JEV.CONTROLLER_WEIGHTS is required for non-off JEV mode'
                )
            controller = None
            if self.jev_mode != 'off':
                controller = build_controller_from_checkpoint(
                    self.jev_controller_weights, device=self.device
                )
            if self.jev_trace_path:
                self.jev_trace_writer = DecisionTraceWriter(self.jev_trace_path)
            self.jev_policy = JEVRuntimePolicy(
                self.jev_mode, controller, self.jev_trace_writer
            )


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        ret['with_attention'] = cfg.MODEL.ASSO_HEAD.WITH_ATTENTION
        ret['thred'] = cfg.MODEL.ASSO_HEAD.THRED
        ret['bank_size'] = cfg.MODEL.ASSO_HEAD.BANK_SIZE
        ret['with_bank'] = cfg.MODEL.ASSO_HEAD.WITH_BANK
        ret['multi_modal'] = cfg.MULTI_MODAL
        ret['jev_enabled'] = cfg.MODEL.JEV.ENABLED
        ret['jev_mode'] = cfg.MODEL.JEV.MODE
        ret['jev_state_dim'] = cfg.MODEL.JEV.STATE_DIM
        ret['jev_max_reassociate'] = cfg.MODEL.JEV.MAX_REASSOCIATE
        ret['jev_trace_path'] = cfg.MODEL.JEV.TRACE_PATH
        ret['jev_controller_weights'] = cfg.MODEL.JEV.CONTROLLER_WEIGHTS
        ret['jev_candidate_enabled'] = cfg.MODEL.JEV.CANDIDATE_ENABLED
        ret['jev_candidate_policy'] = cfg.MODEL.JEV.CANDIDATE_POLICY
        ret['jev_candidate_scripted_model'] = cfg.MODEL.JEV.CANDIDATE_SCRIPTED_MODEL
        return ret

    def _jev_state(self, values):
        """Encode finite online-only evidence for one typed decision."""
        device = getattr(self, 'device', None)
        if device is None:
            device = next(self.parameters()).device
        return encode_state(values, self.jev_state_dim).to(device)

    def _jev_decide(self, state_values, question, legal_actions, off_action, context=None):
        """Return the committed semantic action, preserving OFF semantics."""
        if self.jev_policy is None:
            return off_action
        trace_context = dict(self._jev_context)
        if context:
            trace_context.update(context)
        decision = self.jev_policy.decide(
            self._jev_state(state_values),
            question,
            legal_actions,
            off_action=off_action,
            context=trace_context,
        )
        return decision.committed_action

    def _match_state_values(
        self,
        *,
        accept_score,
        reassociate_score,
        threshold,
        candidate_count,
        candidate_entropy,
        track_count,
        track_age,
        frame_index,
        window_length,
        view_index,
        has_old_track=False,
        current_is_unmatched=False,
        memory_count=0,
        track_score=0.0,
        track_length=1.0,
        score_variance=0.0,
    ):
        """Delegate JEV feature semantics to the canonical shared builder."""

        return build_state_values(
            accept_score=accept_score,
            reassociate_score=reassociate_score,
            threshold=threshold,
            candidate_count=candidate_count,
            candidate_entropy=candidate_entropy,
            track_count=track_count,
            track_age=track_age,
            frame_index=frame_index,
            window_length=window_length,
            view_index=view_index,
            can_reassociate=self.jev_max_reassociate > 0,
            memory_enabled=self.with_bank,
            with_iou=self.with_iou,
            not_mult_thresh=self.not_mult_thresh,
            has_old_track=has_old_track,
            current_is_unmatched=current_is_unmatched,
            memory_count=memory_count,
            track_score=track_score,
            track_length=track_length,
            score_variance=score_variance,
        )

    @staticmethod
    def _jev_tracker_state(
        *,
        id_count=0,
        id_count_dict=None,
        id_reid_dict=None,
        active_ids=None,
        memory_ids=None,
        possible_memory_ids=None,
        stale_ids=None,
        old_reid_count=0,
    ):
        """Serialize the mutable GMT containers at a decision boundary.

        This snapshot is deliberately JSON-sized: it records container
        membership, hit counts, memory lengths and stale-bank membership, but
        never future GT or evaluator output.  Offline branch runners use it
        to restore the same pre-action tracker state before applying each
        typed action.
        """
        id_count_dict = id_count_dict or {}
        id_reid_dict = id_reid_dict or {}
        active = sorted({int(value) for value in (active_ids or [])})
        if possible_memory_ids is None:
            possible_memory_ids = poss_ids.poss_ids
        if stale_ids is None:
            stale_ids = old_ids.old_ids
        memories = {
            str(int(key)): int(len(value))
            for key, value in id_reid_dict.items()
        }
        hits = {str(int(key)): int(value) for key, value in id_count_dict.items()}
        return {
            'id_count': int(id_count),
            'active_track_ids': active,
            'track_hits': hits,
            'memory_lengths': memories,
            'memory_track_ids': sorted(
                {int(value) for value in (memory_ids or [])}
            ),
            'possible_memory_ids': sorted(int(value) for value in (possible_memory_ids or [])),
            'stale_ids': sorted(int(value) for value in (stale_ids or [])),
            'old_reid_count': int(old_reid_count),
        }

    def _candidate_match_values(self, original, scores, ids, match_i, match_j,
                               threshold, *, view, frame_index, window_length,
                               track_lengths, tracker_state, galleries=None,
                               observations=None, view_fractions=None):
        """Production scoring -> shared global assignment; no state commit here."""
        from ..jev_candidate_features import CandidateBatch, evidence12
        from ..jev_candidate_assignment import assign_candidate_values
        profiler = getattr(self, 'jev_candidate_latency_observer', None)
        if profiler is not None:
            torch.cuda.synchronize(scores.device)
            feature_begin = time.perf_counter()
        references = tuple(int(t) for t in ids.tolist())
        m, n = scores.shape
        lengths = torch.ones(n, device=scores.device) if track_lengths is None else torch.as_tensor(track_lengths, device=scores.device)
        pairs = {int(r): int(c) for r, c in zip(match_i, match_j)}
        state = tracker_state or {}
        thresholds = scores.new_tensor([legacy_acceptance_threshold(threshold, float(v), self.not_mult_thresh) for v in lengths])
        legal = torch.isfinite(scores)
        states = []
        for r in range(m):
            clean = torch.where(legal[r], scores[r], scores.new_zeros(n))
            c = pairs.get(r)
            a = float(clean[c]) if c is not None else 0.
            order = torch.argsort(clean, descending=True).tolist()
            other = next((j for j in order if j != c), None)
            b = float(clean[other]) if other is not None else 0.
            probabilities = torch.softmax(clean, dim=0)
            entropy = float(-(probabilities * probabilities.clamp_min(1e-8).log()).sum())
            states.append(self._jev_state(self._match_state_values(
                accept_score=a, reassociate_score=b, threshold=threshold,
                candidate_count=n, candidate_entropy=entropy, track_count=n,
                track_age=0, frame_index=frame_index, window_length=window_length,
                view_index=view, current_is_unmatched=original[r].item() < 0,
                track_score=a, track_length=float(lengths[c]) if c is not None else 1.,
                score_variance=float(clean.var()) if n > 1 else 0.)).to(scores.device))
        state64 = torch.stack(states) if states else scores.new_zeros((0, 64))
        gallery_means = {int(t): value.reid_features.mean(dim=0) for t, value in (galleries or {}).items() if int(t) in references and len(value)}
        features = evidence12(scores, references, pairs, lengths,
            hits={int(t): v for t, v in state.get('track_hits', {}).items()},
            memory_lengths={int(t): v for t, v in state.get('memory_lengths', {}).items()},
            galleries=gallery_means, observations=observations,
            view_fractions=view_fractions, bank_eligible=state.get('possible_memory_ids', ()))
        batch = CandidateBatch(references, scores, state64, features, legal, thresholds, int(view))
        if profiler is not None:
            torch.cuda.synchronize(scores.device)
            policy_begin = time.perf_counter()
            allocated = torch.cuda.memory_allocated(scores.device)
            torch.cuda.reset_peak_memory_stats(scores.device)
        values, newborn = self.jev_candidate_policy.score(batch)
        assignment = assign_candidate_values(values, newborn, references, legal,
            mode=self.jev_candidate_policy.assignment_mode, legacy_thresholds=thresholds)
        if profiler is not None:
            torch.cuda.synchronize(scores.device)
            policy_end = time.perf_counter()
            profiler(batch=batch, feature_ms=(policy_begin-feature_begin)*1000,
                policy_assignment_ms=(policy_end-policy_begin)*1000,
                peak_temporary_bytes=torch.cuda.max_memory_allocated(scores.device)-allocated)
        self._jev_candidate_new_rows = assignment.new_rows if assignment.semantic_new else ()
        self._jev_candidate_last = {'candidate_ids': references, 'batch': batch,
            'assignment': assignment, 'candidate_values': values.detach(), 'new_values': newborn.detach(),
            'view': int(view), 'frame': int(frame_index)}
        return original.new_tensor(assignment.existing_ids)

    def _candidate_observe_commit(self, instances, id_count, hits, galleries,
                                  *, view, frame_index, first=False):
        observer = getattr(self, 'jev_candidate_commit_observer', None)
        if observer is not None:
            observer(instances=instances, id_count=id_count, hits=hits, galleries=galleries,
                possible_ids=poss_ids.poss_ids, old_ids=old_ids.old_ids,
                old_reids=old_reids.old_reids, trajectory_rng=self._jev_trajectory_rng,
                view=int(view), frame=int(frame_index), first=bool(first),
                candidate=getattr(self, '_jev_candidate_last', None))

    def _candidate_initial_pairs(self, scores):
        if getattr(self, 'jev_candidate_policy', None) is None:
            return linear_sum_assignment((-scores).cpu())
        # Invalid current edges must not reach the legacy scipy call before
        # the opt-in value policy gets its finite legal mask.
        pairs = constrained_hungarian(scores)
        return ([r for r, c in pairs], [c for r, c in pairs])

    def _apply_jev_match_decisions(
        self,
        track_ids,
        traj_score,
        unique_ids,
        match_i,
        match_j,
        threshold,
        *,
        view=0,
        frame_index=0,
        window_length=1,
        detection_boxes=None,
        detection_scores=None,
        detection_image_size=None,
        tracker_state=None,
        track_lengths=None,
        candidate_galleries=None,
        candidate_observations=None,
        candidate_view_fractions=None,
    ):
        """Apply typed actions around one global GMT assignment proposal.

        REASSOCIATE rejects the current edge, masks all action-incompatible
        edges, and triggers exactly one full constrained Hungarian solve.  It
        never selects an identity by scanning the second-ranked candidates.
        """
        if getattr(self, 'jev_candidate_policy', None) is not None:
            result = self._candidate_match_values(track_ids, traj_score, unique_ids,
                match_i, match_j, threshold, view=view, frame_index=frame_index,
                window_length=window_length, track_lengths=track_lengths,
                tracker_state=tracker_state, galleries=candidate_galleries,
                observations=candidate_observations, view_fractions=candidate_view_fractions)
            intervention = getattr(self, 'jev_native_match_override', None)
            if intervention is not None and tuple(intervention['key']) == (
                    int(self._jev_context['video_id']), int(frame_index), int(view)):
                from ..jev_native_intervention import apply_native_intervention
                return apply_native_intervention(self, intervention, track_ids,
                    traj_score, unique_ids, match_i, match_j, threshold,
                    view=view, frame_index=frame_index, window_length=window_length,
                    detection_boxes=detection_boxes, detection_scores=detection_scores,
                    detection_image_size=detection_image_size, tracker_state=tracker_state,
                    track_lengths=track_lengths, candidate_galleries=candidate_galleries,
                    candidate_observations=candidate_observations,
                    candidate_view_fractions=candidate_view_fractions)
            return result
        if self.jev_policy is None:
            return track_ids
        n_k = int(traj_score.shape[0])
        n_tracks = int(traj_score.shape[1])
        pair_by_row = {int(i): int(j) for i, j in zip(match_i, match_j)}
        original = track_ids.clone()
        # OFF is an instrumentation mode, not a second assignment
        # implementation.  It still emits typed traces, but must return the
        # exact IDs produced by the original GMT threshold branch so the
        # native-OFF equivalence gate can compare a real golden stream.
        preserve_off_ids = (
            self.jev_policy is not None and getattr(self.jev_policy, 'mode', '') == 'off'
        )
        if track_lengths is None:
            track_lengths = torch.ones(
                n_tracks, dtype=traj_score.dtype, device=traj_score.device
            )
        else:
            track_lengths = torch.as_tensor(
                track_lengths, dtype=traj_score.dtype, device=traj_score.device
            ).reshape(-1)
            if track_lengths.numel() != n_tracks:
                raise ValueError(
                    "track_lengths must have one value per unique track: "
                    f"{track_lengths.numel()} != {n_tracks}"
                )

        first_actions = {}
        accepted_rows = {}
        reassociate_rows = []
        for row in range(n_k):
            row_context = {
                'decision_scope': 'match',
                'detection_index': int(row),
                'proposal_track_id': None,
                'alternate_track_id': None,
                'frame': int(frame_index),
                'view': int(view),
            }
            if detection_boxes is not None and row < len(detection_boxes):
                row_context['bbox_xyxy'] = [
                    float(value) for value in detection_boxes[row].detach().cpu().tolist()
                ]
            if detection_scores is not None and row < len(detection_scores):
                row_context['detection_score'] = float(detection_scores[row].detach().cpu().item())
            if detection_image_size is not None:
                row_context['model_image_size'] = [
                    int(detection_image_size[0]), int(detection_image_size[1])
                ]
            if tracker_state is not None:
                row_context['tracker_state_before'] = copy.deepcopy(tracker_state)
            first_j = pair_by_row.get(row)
            if first_j is None:
                first_actions[row] = 'START_NEW'
                self._jev_decide(
                    self._match_state_values(
                        accept_score=0.0,
                        reassociate_score=0.0,
                        threshold=threshold,
                        candidate_count=0,
                        candidate_entropy=0.0,
                        track_count=len(unique_ids),
                        track_age=0,
                        frame_index=frame_index,
                        window_length=window_length,
                        view_index=view,
                        current_is_unmatched=True,
                    ),
                    'MATCH_DECISION',
                    ['START_NEW'],
                    'START_NEW',
                    context=row_context,
                )
                continue
            row_scores = traj_score[row]
            order = torch.argsort(row_scores, descending=True).tolist()
            first_id = int(unique_ids[first_j].item())
            first_score = float(row_scores[first_j].item())
            second_j = next(
                (j for j in order if int(unique_ids[j].item()) != first_id), None
            )
            second_score = float(row_scores[second_j].item()) if second_j is not None else 0.0
            row_context['candidate_track_ids'] = [
                int(unique_ids[index].item()) for index in order
            ]
            row_context['candidate_scores'] = [
                float(row_scores[index].item()) for index in order
            ]
            entropy = 0.0
            probabilities = torch.softmax(row_scores, dim=0)
            entropy = float((-(probabilities * (probabilities.clamp_min(1e-8).log())).sum()).item())
            off_action = 'ACCEPT_CURRENT' if original[row].item() >= 0 else 'START_NEW'
            legal = ['ACCEPT_CURRENT', 'START_NEW']
            if self.jev_max_reassociate > 0 and second_j is not None:
                legal.insert(1, 'REASSOCIATE')
            action = self._jev_decide(
                self._match_state_values(
                    accept_score=first_score,
                    reassociate_score=second_score,
                    threshold=threshold,
                    candidate_count=len(unique_ids),
                    candidate_entropy=entropy,
                    track_count=len(unique_ids),
                    track_age=0,
                    frame_index=frame_index,
                    window_length=window_length,
                    view_index=view,
                    current_is_unmatched=original[row].item() < 0,
                    memory_count=0,
                    track_score=first_score,
                    track_length=float(track_lengths[first_j].item()),
                    score_variance=float(row_scores.var().item()) if row_scores.numel() > 1 else 0.0,
                ),
                'MATCH_DECISION',
                legal,
                off_action,
                context={
                    **row_context,
                    'proposal_track_id': first_id,
                    'alternate_track_id': int(unique_ids[second_j].item()) if second_j is not None else None,
                },
            )
            first_actions[row] = action
            if action == 'ACCEPT_CURRENT':
                accepted_rows[row] = first_j
            elif action == 'REASSOCIATE' and second_j is not None:
                reassociate_rows.append(row)

        if preserve_off_ids:
            return original

        result = track_ids.new_full((n_k,), -1)
        for row, col in accepted_rows.items():
            result[row] = unique_ids[col]
        if not reassociate_rows:
            return result

        # Only rejected proposal edges are masked.  The solve receives the
        # complete matrix, so an accepted row can move when another row's
        # reassociation changes the globally optimal matching.  START_NEW is
        # the one explicit no-edge constraint carried into round two.
        banned_edges = set()
        for row in range(n_k):
            if first_actions.get(row) == 'START_NEW':
                banned_edges.update((row, col) for col in range(n_tracks))
        for row in reassociate_rows:
            first_j = pair_by_row[row]
            banned_edges.add((row, first_j))

        second_pairs = constrained_hungarian(traj_score, banned_edges)
        second_by_row = {row: col for row, col in second_pairs}
        second_rows = [
            row for row, action in first_actions.items()
            if action in {'ACCEPT_CURRENT', 'REASSOCIATE'}
        ]
        result = track_ids.new_full((n_k,), -1)
        used = set()
        for row in second_rows:
            second_j = second_by_row.get(row)
            if second_j is None:
                continue
            candidate_id = int(unique_ids[second_j].item())
            candidate_score = float(traj_score[row, second_j].item())
            candidate_length = (
                float(track_lengths[second_j].item())
                if track_lengths is not None and second_j < track_lengths.numel()
                else 1.0
            )
            second_off = (
                'ACCEPT_CURRENT'
                if candidate_score > legacy_acceptance_threshold(
                    threshold, candidate_length, self.not_mult_thresh
                )
                else 'START_NEW'
            )
            second_action = self._jev_decide(
                self._match_state_values(
                    accept_score=candidate_score,
                    reassociate_score=0.0,
                    threshold=threshold,
                    candidate_count=len(unique_ids),
                    candidate_entropy=0.0,
                    track_count=len(unique_ids),
                    track_age=0,
                    frame_index=frame_index,
                    window_length=window_length,
                    view_index=view,
                    current_is_unmatched=(
                        first_actions[row] == 'REASSOCIATE' or original[row].item() < 0
                    ),
                    track_score=candidate_score,
                    track_length=float(track_lengths[second_j].item()),
                    score_variance=float(traj_score[row].var().item())
                    if traj_score[row].numel() > 1 else 0.0,
                ),
                'MATCH_DECISION',
                ['ACCEPT_CURRENT', 'START_NEW'],
                second_off,
                context={
                    'decision_scope': 'match_reassociate_validation',
                    'proposal_round': 2,
                    'constrained_hungarian': True,
                    'detection_index': int(row),
                    'proposal_track_id': candidate_id,
                    'rejected_track_id': (
                        int(unique_ids[pair_by_row[row]].item())
                        if first_actions[row] == 'REASSOCIATE' else None
                    ),
                },
            )
            if second_action == 'ACCEPT_CURRENT' and candidate_id not in used:
                result[row] = unique_ids[second_j]
                used.add(candidate_id)
        return original if preserve_off_ids else result

    def _jev_memory_action(self, *, score, threshold, track_count, memory_count, view, frame_index, window_length, track_id=None, detection_index=None, bbox=None, tracker_state=None):
        memory_context = {
            'decision_scope': 'memory',
            'track_id': int(track_id) if track_id is not None else None,
            'detection_index': int(detection_index) if detection_index is not None else None,
        }
        if bbox is not None:
            memory_context['bbox_xyxy'] = [float(value) for value in bbox.detach().cpu().tolist()]
        if tracker_state is not None:
            memory_context['tracker_state_before'] = copy.deepcopy(tracker_state)
        return self._jev_decide(
            self._match_state_values(
                accept_score=score,
                reassociate_score=0.0,
                threshold=threshold,
                candidate_count=1,
                candidate_entropy=0.0,
                track_count=track_count,
                track_age=memory_count,
                frame_index=frame_index,
                window_length=window_length,
                view_index=view,
                memory_count=memory_count,
                track_score=score,
                track_length=max(1.0, float(memory_count)),
            ),
            'MEMORY_DECISION',
            ['WRITE_MEMORY', 'SKIP_MEMORY'],
            'WRITE_MEMORY',
            context=memory_context,
        )

    def _jev_reactivation_action(
        self,
        *,
        score,
        threshold,
        track_count,
        memory_count,
        view,
        frame_index,
        window_length,
        track_id=None,
        detection_index=None,
        bbox=None,
        tracker_state=None,
        candidate_track_ids=None,
        candidate_scores=None,
        bank_threshold=None,
    ):
        reactivate_context = {
            'decision_scope': 'reactivation',
            'track_id': int(track_id) if track_id is not None else None,
            'detection_index': int(detection_index) if detection_index is not None else None,
            # These fields are diagnostic only.  They expose the native GMT
            # stale-bank proposal so the mutable replay can be compared to
            # the same OFF trajectory without using GT or future frames.
            'native_candidate_track_ids': [
                int(value) for value in (candidate_track_ids or [])
            ],
            'native_candidate_scores': [
                float(value) for value in (candidate_scores or [])
            ],
            'native_candidate_order': 'torch_unique_sorted',
            'native_candidate_count': int(len(candidate_track_ids or [])),
            'native_bank_threshold': (
                float(bank_threshold) if bank_threshold is not None else float(threshold)
            ),
            'native_bank_threshold_base': float(self.thred_bank),
        }
        if bbox is not None:
            reactivate_context['bbox_xyxy'] = [float(value) for value in bbox.detach().cpu().tolist()]
        if tracker_state is not None:
            reactivate_context['tracker_state_before'] = copy.deepcopy(tracker_state)
        return self._jev_decide(
            self._match_state_values(
                accept_score=0.0,
                reassociate_score=score,
                threshold=threshold,
                candidate_count=1,
                candidate_entropy=0.0,
                track_count=track_count,
                track_age=memory_count,
                frame_index=frame_index,
                window_length=window_length,
                view_index=view,
                has_old_track=True,
                current_is_unmatched=True,
                memory_count=memory_count,
                track_score=score,
                track_length=max(1.0, float(memory_count)),
            ),
            'REACTIVATION_DECISION',
            ['REACTIVATE_OLD', 'START_NEW'],
            'REACTIVATE_OLD' if score > threshold else 'START_NEW',
            context=reactivate_context,
        )


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        min = 999999
        max = -99999
        frame_num = []
        for num,i in enumerate(batched_inputs):
            num = int(i['file_name'].split('.')[0][-6:])

            #mvmhat
            #num = int(i['file_name'].split('.')[0].split('/')[-1])
            if num>max:
                max = num
            if num<min:
                min = num
            frame_num.append(num)
        view_num = batched_inputs[0]['view_num']
        if not self.training:
            if self.local_track:
                return self.local_tracker_inference(batched_inputs)
            return self.sliding_inference_GMT(batched_inputs, view_num, [min, max, frame_num])

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs ]
        if self.multi_modal:
            features = self.fuse_feature(features)
            gt_instances = [x["instances"].to(self.device) for i,x in enumerate(batched_inputs) if i%2==0 ]
            images = ImageList(images.tensor[range(0,len(images),2)],images.image_sizes[0:len(images):2])
        try:
            view_num = batched_inputs[0]["view_num"]
        except :
            view_num = -1

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        _, detector_losses = self.roi_heads(
            images, features, proposals,view_num,[min,max,frame_num],None, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        return losses

               

    def sliding_inference(self, batched_inputs):
        if getattr(self, 'jev_candidate_policy', None) is not None:
            raise NotImplementedError('candidate values require the native sliding_inference_GMT entry')
        video_len = len(batched_inputs)
        instances = []
        id_count = 0
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])

            if frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            else:
                win_st = max(0, frame_id + 1 - self.test_len)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self.run_global_tracker(
                    batched_inputs[win_st: win_ed],
                    instances[win_st: win_ed],
                    k=min(self.test_len - 1, frame_id),
                    id_count=id_count) # n_k x N
            if frame_id - self.test_len >= 0:
                instances[frame_id - self.test_len].remove(
                    'reid_features')

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances
    
    def sliding_inference_GMT(self, batched_inputs,view_num,time, *, native_prefix=None, native_stop_frame=None, native_raw=False):
        poss_ids.poss_ids = set()
        old_ids.old_ids = set()
        old_reids.old_reids = []
        if self.jev_enabled:
            from ..roi_heads.transformer import TrajectoryRandom

            master_seed = int(os.environ.get('JEV_TRAJECTORY_RNG_MASTER_SEED', '20261006'))
            video_id = int(batched_inputs[0].get('video_id', -1))
            self._jev_trajectory_rng = TrajectoryRandom(master_seed + video_id)
        else:
            self._jev_trajectory_rng = None
        self._jev_context = {
            'video_id': int(batched_inputs[0].get('video_id', -1)),
            'view_num': int(view_num),
        }
        view_num = batched_inputs[0]['view_num']
        view_frames = int(len(batched_inputs)/view_num)
        instances = []
        id_count = 0
        id_count_dict = dict()
        id_reid_dict = dict()
        memory_bank = []
        start_frame = 0
        resume_view = None
        if native_prefix is not None:
            from ..jev_native_state import restore_state
            restore_state(self, native_prefix)
            assert native_prefix['key'][0] == int(batched_inputs[0]['video_id'])
            instances = native_prefix['instances']
            id_count = native_prefix['id_count']
            id_count_dict = native_prefix['hits']
            id_reid_dict = native_prefix['galleries']
            start_frame, resume_view = native_prefix['key'][1:]
        end_frame = view_frames if native_stop_frame is None else min(view_frames, native_stop_frame + 1)
        prefix_observer = getattr(self, 'jev_native_prefix_observer', None)
        for frame_id in tqdm(range(start_frame, end_frame)):
            self._jev_context['frame'] = int(frame_id)
            batched_inputs_divo = []
            st = frame_id
            instances_wo_id = []
            time_per = time.copy()
            for view in range(0 if frame_id != start_frame or native_prefix is None else view_num, view_num):
                self._jev_context['view'] = int(view)
                time_per[2] = time[2][st]
                if self.jev_perception_cache_reader is not None:
                    cached = self.jev_perception_cache_reader.load(
                        int(batched_inputs[st].get('video_id', -1)),
                        int(frame_id),
                        int(view),
                    )
                    cached_instances = Instances(
                        tuple(int(value) for value in cached['image_size'])
                    )
                    cached_instances.pred_boxes = Boxes(
                        torch.as_tensor(
                            cached['pred_boxes'],
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                    cached_instances.scores = torch.as_tensor(
                        cached['detection_scores'],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # The immutable cache stores proposal class labels in
                    # ``proposal_metadata`` rather than in the tensor payload.
                    # Native GMT returns ``pred_classes`` and the evaluator
                    # requires that field even when the tracker itself only
                    # uses boxes/scores/ReID.  Restore it here so a
                    # cache-backed native replay has the same output schema
                    # as the detector-backed path; fail closed if an older or
                    # incomplete cache cannot provide the labels.
                    proposal_metadata = cached.get('proposal_metadata', {})
                    if 'pred_classes' not in proposal_metadata:
                        raise KeyError(
                            'perception cache record is missing proposal_metadata.pred_classes'
                        )
                    cached_instances.pred_classes = torch.as_tensor(
                        proposal_metadata['pred_classes'],
                        dtype=torch.long,
                        device=self.device,
                    )
                    cached_instances.reid_features = torch.as_tensor(
                        cached['reid_features'],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    instances_wo_id.append(cached_instances)
                else:
                    instances_wo_id += self.inference(
                        batched_inputs[st: st + 1],
                        view_num,
                        time_per,
                        view,
                        do_postprocess=False)
                if self.jev_perception_cache is not None:
                    cache_input = batched_inputs[st]
                    self.jev_perception_cache.write(
                        video_id=int(cache_input.get('video_id', -1)),
                        frame=int(frame_id),
                        view=int(view),
                        instances=instances_wo_id[-1],
                        metadata={
                            'file_name': str(cache_input.get('file_name', '')),
                            'dataset_frame': int(time_per[2])
                            if isinstance(time_per[2], (int, np.integer)) else str(time_per[2]),
                            'view_num': int(view_num),
                        },
                    )
                st += view_frames
            instances.extend([x for x in instances_wo_id])
            activate_first = True
            activate = True

            if frame_id == 0:
                first_frame_num = [len(instances[i])  for i in range(view_num)]
                sort_index = np.argsort(first_frame_num)
                sort_index = sort_index.tolist()
                sort_index.reverse()
                max_index = sort_index[0]

                if native_prefix is None:
                    instances[max_index].track_ids = torch.arange(
                        1, len(instances[max_index]) + 1,
                        device=instances[max_index].reid_features.device)
                    id_count = len(instances[max_index])
                    for i in range(1, len(instances[max_index]) + 1):
                        id_count_dict[i] = 1
                        id_reid_dict[i] = instances[max_index][i-1]
                #id = ([i for i in range(view_num)])
                sort_index.remove(max_index)
                id = np.sort(sort_index)
                instances_kv = [instances[max_index]]
                if activate_first :
                    x = []

                    for i in range(view_num-1):
                        x = [instances[id[i]]]
                        a = Instances(instances[0].image_size)
                        a = a.cat(x)
                        instances_kv += [a]
                        if native_prefix is not None and frame_id == start_frame and int(id[i]) < resume_view:
                            continue
                        if prefix_observer is not None:
                            prefix_observer(instances=instances, id_count=id_count, hits=id_count_dict,
                                galleries=id_reid_dict, frame=frame_id, view=int(id[i]), first=True)
                        asso_output, pred_boxes, n_t, Np, query_inds = self.get_asso(
                            instances_kv,
                            k=len(instances_kv) - 1)  # n_k x N

                        instances_kv =instances_kv
                        instances_kv, id_count,id_count_dict,id_reid_dict  = self.run_first_tracker_plus(
                            instances_kv,
                            [asso_output[0][:,:Np]],
                            pred_boxes[:Np,:],
                            len(instances_kv)-1,
                            id_count,
                            id_count_dict,
                            id_reid_dict,
                            view=id[i],
                            frame_index=frame_id)
                        #start = end
                        instances[id[i]] = instances_kv[len(instances_kv)-1]
                else:
                   # instances_kv = [instances[max_index]]
                    for i in range(view_num-1):
                        instances_kv = instances_kv + [instances[id[i]]]
                        #instances_kv = instances_kv + [instances[i+1]]
                        instances_kv, id_count = self.run_first_tracker(
                            batched_inputs[: 2],
                            instances_kv,
                            k=len(instances_kv)-1,
                            id_count=id_count) # n_k x N
                        instances[id[i]] = instances_kv[len(instances_kv)-1]
                        #query += 1
                    id_count = 0
                    for i in range(view_num):
                        id_count = max(id_count,(max(instances[i].track_ids )).item())
            else:
                win_st = max(0, frame_id + 1 - self.test_len)*view_num
                win_ed = view_num*frame_id
               # activate  = True
                instances_kv = instances[win_st:win_ed]
                if  activate:
                    instacnes_old = native_prefix['frame_old_instances'] if native_prefix is not None and frame_id == start_frame else instances[:win_st]
                    for i in range(view_num):
                        instances_kv = instances_kv + [instances[win_ed+i]]
                        if native_prefix is not None and frame_id == start_frame and i < resume_view:
                            continue
                        if prefix_observer is not None:
                            prefix_observer(instances=instances, id_count=id_count, hits=id_count_dict,
                                galleries=id_reid_dict, frame=frame_id, view=i, first=False,
                                frame_old_instances=instacnes_old)
                        asso_output, pred_boxes, n_t, Np, query_inds = self.get_asso(
                            instances_kv,
                            k=len(instances_kv) - 1)

                        instances_kv, id_count,id_count_dict = self.run_global_tracker_plus(
                            view_num,
                            instances_kv,
                            [asso_output[0][:,:Np]],
                            pred_boxes[:Np,:],
                            len(instances_kv)-1,
                            id_count,
                            id_count_dict,
                            id_reid_dict,
                            instacnes_old,
                            i,
                            frame_index=frame_id)
                        instances[win_ed+i] = instances_kv[-1]
                else :
                    for i in range(view_num):
                        instances_kv =instances_kv + [instances[win_ed+i]]
                        instances_kv, id_count = self.run_global_tracker(
                            batched_inputs[win_st: win_ed],
                            instances_kv,
                            k=len(instances_kv)-1,
                            id_count=id_count) # n_k x N
                        instances[win_ed+i] = instances_kv[len(instances_kv)-1]


        if native_raw:
            return instances, view_num
        batch = []
        #调整batch的顺序，和instances一致，view1_frame1,view_2_frame1,view_3_frame1 
        for i in range(view_frames):
            for j in range(view_num):
                batch.append(batched_inputs[j*view_frames+i])
        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batch, [
                    (0, 0) for _ in range(len(batch))],
                not_clamp_box=self.not_clamp_box)
        for i in range(len(batch)):
            batch[i]['image'] = None        
        return instances,view_num

    def run_first_tracker_plus(
        self,
        instances,
        asso_output,
        pred_boxes,
        k,
        id_count,
        id_count_dict,
        id_reid_dict,
        *,
        view=0,
        frame_index=0,
    ):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        asso_nonk = self.roi_heads._activate_asso(asso_output)[0]

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np

        unique_ids = torch.unique(ids) # M
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = self._candidate_initial_pairs(traj_score) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = legacy_acceptance_threshold(
                self.overlap_thresh,
                float(id_inds[:, j].sum().item()),
                self.not_mult_thresh,
            )
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        if self.jev_policy is not None or getattr(self, 'jev_candidate_policy', None) is not None:
            self._jev_context['view'] = int(view)
            track_ids = self._apply_jev_match_decisions(
                track_ids,
                traj_score,
                unique_ids,
                match_i,
                match_j,
                self.overlap_thresh,
                view=view,
                frame_index=frame_index,
                window_length=max(1, T),
                detection_boxes=instances[k].pred_boxes.tensor,
                detection_scores=instances[k].scores if instances[k].has('scores') else instances[k].objectness_logits,
                detection_image_size=instances[k].image_size,
                tracker_state=self._jev_tracker_state(
                    id_count=id_count,
                    id_count_dict=id_count_dict,
                    id_reid_dict=id_reid_dict,
                    active_ids=unique_ids.tolist(),
                    memory_ids=poss_ids.poss_ids,
                ),
                track_lengths=id_inds.sum(dim=0),
                candidate_galleries=id_reid_dict,
                candidate_observations=instances[k].reid_features,
            )

        # Keep every current-frame slice field-compatible with the historical
        # Instances stored in id_reid_dict before any memory concatenation.
        instances[k].track_ids = track_ids
        for i in range(n_k):
            id =  track_ids[i].item()
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
                id_count_dict[id_count] = 1
                instances[k].track_ids = track_ids#修改
                id_reid_dict[id_count] = instances[k][i]
            else :
                id_count_dict[id] += 1
                instances[k].track_ids = track_ids#修改
                memory_action = self._jev_memory_action(
                    score=float(traj_score[i].max().item()) if traj_score.shape[1] else 0.0,
                    threshold=self.overlap_thresh,
                    track_count=len(unique_ids),
                    memory_count=len(id_reid_dict[id]),
                    view=view,
                    frame_index=frame_index,
                    window_length=max(1, T),
                    track_id=id,
                    detection_index=i,
                    bbox=instances[k].pred_boxes.tensor[i],
                    tracker_state=self._jev_tracker_state(
                        id_count=id_count,
                        id_count_dict=id_count_dict,
                        id_reid_dict=id_reid_dict,
                        active_ids=unique_ids.tolist(),
                        memory_ids=poss_ids.poss_ids,
                    ),
                )
                if memory_action == 'WRITE_MEMORY':
                    instance_cat = [id_reid_dict[id] , instances[k][i]]
                    id_reid_dict[id] = Instances.cat(instance_cat)
                #id_reid_dict[id].reid_features = id_reid_dict[id].reid_features/id_count_dict[id]*(id_count_dict[id]-1)+instances[k][i].reid_features/id_count_dict[id]
        
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        self._candidate_observe_commit(instances, id_count, id_count_dict, id_reid_dict,
            view=view, frame_index=frame_index, first=True)
        return instances, id_count ,id_count_dict,id_reid_dict

    def get_asso(self,  instances, k,):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        n_k = len(instances[k])
        Np = N - n_k
        traj_ids=torch.Tensor().cuda()
        for i in range(len(instances)-1):
            traj_ids=torch.cat([traj_ids,instances[i].track_ids])
        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        transformer_kwargs = {}
        if self._jev_trajectory_rng is not None:
            transformer_kwargs['trajectory_rng'] = self._jev_trajectory_rng
        asso_output, pred_boxes, _, query_inds,_,_ = self.roi_heads._forward_transformer(
            instances, reid_features,None, k,None,None,traj_ids,
            **transformer_kwargs) # [n_k x N], N x 4
        return asso_output,pred_boxes,n_t,Np,query_inds

    def get_attention(self,instances,view_num,query_inds,id=None):
        proposals = [x for  x in  instances]
        #features = [features[f] for f in self.asso_in_features]
        proposal_boxes = [x.pred_boxes for x in proposals] # 
        n_t = [len(x) for x in proposals]
        if id==None:
            num_frame = int(len(instances)/view_num)
            spatial_feature = torch.zeros((1,sum(n_t),4))
            for i in range(len(n_t)):
                for j in range(4):
                    if j==0 or j==2:
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[1]
                    else :
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[0]
            camera_feature = torch.zeros((1,sum(n_t),1))
            for i in range(len(n_t)):
                camera_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),0] = (i%view_num)/float(view_num)
            time_featrue = torch.zeros((1,sum(n_t),1))
            for i in range(num_frame):
                time_featrue[0,sum(n_t[:i*view_num]):sum(n_t[:(i+1)*view_num]),0] = float(i)/(num_frame)
            spatial_feature = spatial_feature.to(proposal_boxes[0].device)
            time_featrue = time_featrue.to(proposal_boxes[0].device)
            camera_feature = camera_feature.to(proposal_boxes[0].device)
            inputs=[spatial_feature,time_featrue,camera_feature]
            s_t_feature = torch.cat(inputs,dim=2)
        else :
            num_frame = int(len(instances)/view_num)
            spatial_feature = torch.zeros((1,sum(n_t),4))
            for i in range(len(n_t)):
                for j in range(4):
                    if j==0 or j==2:
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[1]
                    else :
                        spatial_feature[0,sum(n_t[:i]):sum(n_t[:i+1]),j] = proposal_boxes[i].tensor[:,j]/instances[i].image_size[0]
            camera_feature = torch.zeros((1,sum(n_t),1))
            for i in range(view_num):
                camera_feature[0,sum(n_t[0:(i)*num_frame]):sum(n_t[0:(i+1)*num_frame]),0] = i/float(view_num)
            time_featrue = torch.zeros((1,sum(n_t),1))
            for i in range(len(n_t)):
                time_featrue[0,sum(n_t[:i]):sum(n_t[:i+1]),0] = float(i%num_frame)/(num_frame)
            spatial_feature = spatial_feature.to(proposal_boxes[0].device)
            time_featrue = time_featrue.to(proposal_boxes[0].device)
            camera_feature = camera_feature.to(proposal_boxes[0].device)
            inputs=[spatial_feature,time_featrue,camera_feature]
            s_t_feature_temp = torch.cat(inputs,dim=2)
            s_t_feature = torch.zeros_like(s_t_feature_temp)
            start = 0
            for i in range(len(n_t)):
                s_t_feature[0,start:start+n_t[id[i]],:] = s_t_feature_temp[0,sum(n_t[:id[i]]) :sum(n_t[:id[i]+1]),:]
                start += n_t[id[i]]
        return    self.roi_heads._forward_transformer_attention(self.roi_heads.s_t_head(s_t_feature),query_inds)
                
    def run_global_tracker_plus(
        self,
        view_num,
        instances,
        asso_output,
        pred_boxes,
        k,
        id_count,
        id_count_dict,
        id_reid_dict,
        instances_old,
        view,
        *,
        frame_index=None,
    ):
        self._jev_context['view'] = int(view)
        if frame_index is None:
            frame_index = int(k // max(1, view_num))
        else:
            frame_index = int(frame_index)
        self._jev_context['frame'] = frame_index
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        #view_num = 3 #这部分代码要把3替换调
        asso_output = asso_output[-1].split(n_t[:-1], dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_nonk = torch.cat(asso_output, dim=1) # n_k x N
        last_frame_num = 2
        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        nonk_inds_view = []
        nonk_inds_last = []
        start = 0
        end = 0
        for t,x in enumerate(n_t):
            end += x
            if t!=k and (k-t)%view_num ==0:
                nonk_inds_view +=   range(start,end)
            if t!=k and (k-t)%view_num ==0 and (k-t)/view_num<=last_frame_num:
                nonk_inds_last +=   range(start,end)            
            start += x


        unique_ids = torch.unique(ids) # M

        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = self._candidate_initial_pairs(traj_score) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = legacy_acceptance_threshold(
                self.overlap_thresh,
                float(id_inds[:, j].sum().item()),
                self.not_mult_thresh,
            )
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]
        if self.jev_policy is not None or getattr(self, 'jev_candidate_policy', None) is not None:
            track_ids = self._apply_jev_match_decisions(
                track_ids,
                traj_score,
                unique_ids,
                match_i,
                match_j,
                self.overlap_thresh,
                view=view,
                frame_index=frame_index,
                window_length=max(1, T // max(1, view_num)),
                detection_boxes=instances[k].pred_boxes.tensor,
                detection_scores=instances[k].scores if instances[k].has('scores') else instances[k].objectness_logits,
                detection_image_size=instances[k].image_size,
                tracker_state=self._jev_tracker_state(
                    id_count=id_count,
                    id_count_dict=id_count_dict,
                    id_reid_dict=id_reid_dict,
                    active_ids=unique_ids.tolist(),
                    memory_ids=poss_ids.poss_ids,
                ),
                track_lengths=id_inds.sum(dim=0),
                candidate_galleries=id_reid_dict,
                candidate_observations=instances[k].reid_features,
            )
        if self.with_bank:
            flag = False
            #a = Instances(instances[0].image_size)
            isinstances_no_match = [instances[-1]]#   取最后一个待匹配帧
            index = []
            semantic_new = set(getattr(self, '_jev_candidate_new_rows', ())) if getattr(self, 'jev_candidate_policy', None) is not None else set()
            for i in range(n_k):
                if track_ids[i] < 0 and i not in semantic_new:
                    index.append(i)
            isinstances_no_match[0] = isinstances_no_match[0][index]#   取所有没有匹配上的instance
            track_id_memory ,instances_matching,run_time= self.memory_bank(id_count_dict,id_reid_dict,unique_ids,instances_old,isinstances_no_match)
            count = 0
            if track_id_memory!=None:
                for i in range(n_k):
                    if i in index:
                        if track_id_memory[count]>=0:
                            track_ids[i] = track_id_memory[count]
                           # print("************ohhhhhhhhhhhh********")
                            print('id',track_id_memory[count],'view',view)
                            print(run_time)
                        count += 1
        #poss_ids.poss_ids = set()
        # Memory concatenation requires the current slice to expose the same
        # fields as the historical Instances, including track_ids.
        instances[k].track_ids = track_ids
        for i in range(n_k):
            id = track_ids[i].item()
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
                id_count_dict[id_count] = 1
                instances[k].track_ids = track_ids#修改
                id_reid_dict[id_count] = instances[k][i]
            else :
                id_count_dict[id] += 1
                memory_action = self._jev_memory_action(
                    score=float(traj_score[i].max().item()) if traj_score.shape[1] else 0.0,
                    threshold=self.overlap_thresh,
                    track_count=len(unique_ids),
                    memory_count=len(id_reid_dict[id]),
                    view=view,
                    frame_index=frame_index,
                    window_length=max(1, T // max(1, view_num)),
                    track_id=id,
                    detection_index=i,
                    bbox=instances[k].pred_boxes.tensor[i],
                    tracker_state=self._jev_tracker_state(
                        id_count=id_count,
                        id_count_dict=id_count_dict,
                        id_reid_dict=id_reid_dict,
                        active_ids=unique_ids.tolist(),
                        memory_ids=poss_ids.poss_ids,
                    ),
                )
                if memory_action == 'WRITE_MEMORY':
                    instance_cat = [id_reid_dict[id] , instances[k][i]]
                    id_reid_dict[id] = Instances.cat(instance_cat)
                if len(id_reid_dict[id])==self.bank_size+1:
                    poss_ids.poss_ids.add(id)
                instances[k].track_ids = track_ids#修改
                #id_reid_dict[id].reid_features = id_reid_dict[id].reid_features/id_count_dict[id]*(id_count_dict[id]-1)+instances[k][i].reid_features/id_count_dict[id]
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        self._candidate_observe_commit(instances, id_count, id_count_dict, id_reid_dict,
            view=view, frame_index=frame_index)
        return instances, id_count,id_count_dict
 
    def memory_bank(self,id_count_dict,id_reid_dict,unique_ids,instances_old,instances_no_match):
        time1 = time.time()
        if len(instances_old)==0 or len(instances_no_match[0])==0:
            return  None,None,None
        instacnes_to_match = []
        thred = self.bank_size
        instances_old.reverse()
        memory_ids=[]
        for id in poss_ids.poss_ids.copy():
            sum_reid = 0
            if id not in unique_ids and len(id_reid_dict.get(id, [])) >= thred:
                memory_ids.append(id)
                #old_ids.old_ids.add(id)
                poss_ids.poss_ids.remove(id)
                for i in range(thred):
                    sum_reid += id_reid_dict[id][-1-i].reid_features
                aver_reid = sum_reid / thred
                #id_reid_dict[id][0].reid_features = aver_reid
                old_reids.old_reids.append(id_reid_dict[id][0])
                if len(old_reids.old_reids)==1:
                    old_reids.old_reids[0].reid_features = aver_reid
                else:
                    old_reids.old_reids[1].reid_features = aver_reid
                old_reids.old_reids = Instances.cat(old_reids.old_reids)
                old_reids.old_reids = [old_reids.old_reids]
                    
        time2 = time.time()

        if len(old_reids.old_reids)==0:
            return None,None,None

        instances_matching = old_reids.old_reids.copy()
        instances_matching.append(instances_no_match[0])
        time3 = time.time()
        asso_output,pred_boxes,n_t,Np,query_inds= self.get_asso(
            instances_matching,
            k= len(instances_matching)-1) # n_k x N       
        time4 = time.time()
        track_ids = self.run_memory_tracker(instances_matching,asso_output,pred_boxes,len(instances_matching)-1,id_count_dict)
        time5 = time.time()
        return track_ids, instacnes_to_match,[time2-time1,time3-time2,time4-time3,time5-time4]
                
#                if id in instances_old[i]['track_ids']
    def run_memory_tracker(self, instances,asso_output,pred_boxes,k,id_count_dict):    
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)
        asso_nonk = self.roi_heads._activate_asso(asso_output)[0]

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np

        unique_ids = torch.unique(ids) # M
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = legacy_acceptance_threshold(
                self.thred_bank,
                float(id_inds[:, j].sum().item()),
                self.not_mult_thresh,
            )
            # A traced/active JEV controller must see the typed reactivation
            # question even when the legacy gate would reject the stale ID;
            # OFF mode's helper still returns the original threshold action.
            if self.jev_policy is not None or traj_score[i, j] > thresh:
                action = self._jev_reactivation_action(
                    score=float(traj_score[i, j].item()),
                    threshold=thresh,
                    track_count=len(unique_ids),
                    memory_count=int(id_count_dict.get(int(unique_ids[j].item()), 0)),
                    view=0,
                    frame_index=k,
                    window_length=max(1, T),
                    track_id=int(unique_ids[j].item()),
                    detection_index=int(i),
                    bbox=instances[k].pred_boxes.tensor[i],
                    candidate_track_ids=unique_ids.detach().cpu().tolist(),
                    candidate_scores=traj_score[i].detach().cpu().tolist(),
                    bank_threshold=self.thred_bank,
                    tracker_state=self._jev_tracker_state(
                        id_count=max(id_count_dict.keys(), default=0),
                        id_count_dict=id_count_dict,
                        id_reid_dict={},
                        active_ids=unique_ids.tolist(),
                        memory_ids=poss_ids.poss_ids,
                    ),
                )
                if action == 'REACTIVATE_OLD':
                    track_ids[i] = unique_ids[j]
                    # old_ids.old_ids.remove(unique_ids[j].item())
                    poss_ids.poss_ids.add(unique_ids[j].item())
                    a = len(old_reids.old_reids[0])
                    for old_index in range(a):
                        if old_reids.old_reids[0][old_index].track_ids.item() == unique_ids[j].item():
                            old_reids.old_reids = [Instances.cat([old_reids.old_reids[0][:old_index], old_reids.old_reids[0][old_index+1:]])]
                            break

       # assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return track_ids

    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = FairMOT(
            no_iou=self.local_no_iou,
            iou_only=self.local_iou_only)

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets, id_feature)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances
