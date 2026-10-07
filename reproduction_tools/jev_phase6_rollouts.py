"""Exact online pilot transitions with branch snapshots and live future policy.

Instrumentation stays outside the tracker. Counterfactual forks resume the
entire mutable state, including Python trajectory RNG, before the event key.
Every future decision is recomputed; no frozen OFF action map is replayed.
"""
from collections import Counter
import json
import os
from pathlib import Path

from jev_phase6_common import ROOT, OUT, B2, load_controller, protect_anchor, new_output, save


class EndEventHorizon(Exception):
    pass


class NativeReplayLab:
    def __init__(self, video, device='cuda:0'):
        protect_anchor()
        empty = OUT / 'empty_debug_inputs.jsonl'
        if not empty.exists(): empty.write_text('')
        os.environ.update(JEV_VIDEO_ID=str(video), JEV_TRACE_PATH=str(empty), JEV_RECORDS_PATH=str(empty))
        import run_early_pilot_tracking as pilot
        self.pilot = pilot
        pilot.VIDEO_ID = video; pilot.TRACE = pilot.RECORDS = empty
        self.video = video; self.device = device
        names = ['build_formal_gmt_engine','MutableGMTState','FrozenPerceptionCache','JEVRuntimePolicy',
                 'build_controller_from_checkpoint','build_state_features','association_window_length',
                 'candidate_entropy','count_memory_observations','count_track_history','feature_names','legacy_acceptance_threshold']
        self.kwargs = dict(zip(names, pilot.load_runtime_modules())); self.kwargs.pop('feature_names')
        self.engine = self.kwargs['build_formal_gmt_engine'](
            config_file=pilot.CONFIG, checkpoint=pilot.CHECKPOINT, device=device, view_num=2, history_limit=80)
        self.engine.association_fn.model.to(device).eval()
        self.cache = self.kwargs['FrozenPerceptionCache'](pilot.CACHE)
        self.keys, self.seed_key, self.first_counts = pilot.ordered_replay_keys(self.cache, video, 2)
        self.seed_state, self.seed_payload = pilot.seed_production_state(self.cache,self.kwargs['MutableGMTState'],self.seed_key)
        _, self.subset, self.lookup, _, _ = pilot.load_inputs()
        self.controller = load_controller(B2, 'cpu')
        self.kwargs.update(build_formal_gmt_engine=lambda **kw:self.engine,
                           FrozenPerceptionCache=lambda path:self.cache,
                           build_controller_from_checkpoint=lambda path,device='cpu':self.controller)
        self.original_propose = self.engine.propose
        self.original_step = self.engine.step
        self.original_append = pilot.append_predictions
        self.current = {}; self.prefix = None; self.decision_rows = []
        self.before_step = None; self.after_step = None; self.intervention = None
        self.engine.propose = self.propose
        self.engine.step = self.step
        pilot.choose = self.choose

    def propose(self, payload, state, **kwargs):
        if not state.reactivation_mode:
            self.current = payload; self.prefix = state
            self.decision_rows = []
        return self.original_propose(payload, state, **kwargs)

    def choose(self, policy, feature, question, legal, off_action, context=None):
        action = policy.decide(feature,question,legal,off_action=off_action,context=context).committed_action if question=='MATCH_DECISION' else off_action
        original = action
        context = context or {}
        if self.intervention is not None:
            action = self.intervention(feature,question,legal,action,context)
        if action not in legal: raise AssertionError('illegal event intervention')
        self.decision_rows.append({'question':question,'action':action,'original_action':original,
                                   'off_action':off_action,'feature_vector':feature.detach().cpu().tolist(),
                                   'context':context})
        return action

    def step(self, payload, state, **kwargs):
        key = (int(payload['video_id']),int(payload['frame']),int(payload['view']))
        if self.before_step is not None:
            self.before_step(key,payload,state,kwargs,self.decision_rows)
        result = self.original_step(payload,state,**kwargs)
        if self.after_step is not None:
            self.after_step(key,payload,state,result,kwargs,self.decision_rows)
        return result

    def run(self, output, *, prefix=None, start_key=None, end_frame=None,
            intervention=None, before_step=None, after_step=None):
        pilot = self.pilot
        output = new_output(output); pilot.PILOT = output
        self.intervention = intervention; self.before_step = before_step; self.after_step = after_step
        if prefix is None:
            keys = self.keys
            pilot.seed_production_state = lambda cache,cls,key:(self.seed_state.clone(),self.seed_payload)
            pilot.append_predictions = self.original_append
        else:
            index = self.keys.index(tuple(start_key))
            continuation = self.keys[index:]
            if end_frame is not None: continuation = [k for k in continuation if k[1]<=end_frame]
            # The seed descriptor retains the original frame-zero semantics;
            # only state and subsequent perception keys are replaced.
            keys = [self.seed_key] + continuation
            pilot.seed_production_state = lambda cache,cls,key:(prefix.clone(),self.seed_payload)
            skipped = False
            def append(predictions,payload,image,committed):
                nonlocal skipped
                if not skipped: skipped=True; return
                self.original_append(predictions,payload,image,committed)
            pilot.append_predictions = append
        pilot.ordered_replay_keys = lambda cache,video_id,view_num:(keys,self.seed_key,self.first_counts)
        try:
            result = pilot.run_method('jev',B2,image_lookup=self.lookup,by_key={},records={},
                feature_source_mode='runtime',parity_report={},device=self.device,**self.kwargs)
            save(output/'result.json',result)
            return result
        finally:
            self.intervention = self.before_step = self.after_step = None
            pilot.append_predictions = self.original_append


def event_key(row):
    c=row['context']
    return (int(c['video_id']),int(c['frame']),int(c['view']),row['question'],int(c['detection_index']))
