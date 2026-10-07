"""Exact online pilot transitions with branch snapshots and live future policy.

Instrumentation stays outside the tracker. Counterfactual forks resume the
entire mutable state, including Python trajectory RNG, before the event key.
Every future decision is recomputed; no frozen OFF action map is replayed.
"""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import time

from jev_phase6_common import ROOT, OUT, B2, load_controller, protect_anchor, new_output, save


class EndEventHorizon(Exception):
    pass


class EventHorizonKeys(list):
    def __init__(self, keys, lab):
        super().__init__(keys); self.lab=lab

    def __getitem__(self, item):
        value=super().__getitem__(item)
        if isinstance(item,slice) and item.start==1:
            def continuation():
                for key in value:
                    if self.lab.stop_requested: break
                    yield key
            return continuation()
        return value


class NativeReplayLab:
    def __init__(self, video, device='cuda:0',native_match_validation=False,record_transitions=False,fast_match=False):
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
        self.metadata={}
        self.update_metadata(self.seed_payload,self.seed_state,{r:r+1 for r in range(len(self.seed_payload['pred_boxes']))},{})
        self.seed_metadata=copy.deepcopy(self.metadata)
        _, self.subset, self.lookup, _, _ = pilot.load_inputs()
        self.controller = load_controller(B2, 'cpu')
        self.fast_match=None
        if fast_match:
            from jev_phase6_fast_match import FastMatchPolicy
            self.fast_match=FastMatchPolicy(self.controller)
        self.native_match_validation=native_match_validation;self.native_resolver=None
        if native_match_validation:
            from jev_phase6_native_match import NativeMatchResolver
            self.native_resolver=NativeMatchResolver(self.engine,self.controller)
            self.engine.resolve_actions=self.native_resolver.resolve
        self.kwargs.update(build_formal_gmt_engine=lambda **kw:self.engine,
                           FrozenPerceptionCache=lambda path:self.cache,
                           build_controller_from_checkpoint=lambda path,device='cpu':self.controller)
        self.original_propose = self.engine.propose
        self.original_step = self.engine.step
        self.original_append = pilot.append_predictions
        self.current = {}; self.prefix = None; self.decision_rows = []
        self.before_step = None; self.after_step = None; self.intervention = None
        self.stop_requested=False; self.steps_completed=0
        self.status_stamp=0;self.output=None
        self.record_transitions=record_transitions;self.transition_buffer=[];self.journal_files=[]
        self.engine.propose = self.propose
        self.engine.step = self.step
        pilot.choose = self.choose

    def propose(self, payload, state, **kwargs):
        if not state.reactivation_mode:
            self.current = payload; self.prefix = state
            self.decision_rows = []
            if self.native_resolver is not None:self.native_resolver.reset()
            for track,m in self.metadata.items():
                if track not in state.active_ids:m.setdefault('stale_since',int(payload['frame']))
                else:m.pop('stale_since',None)
        return self.original_propose(payload, state, **kwargs)

    def choose(self, policy, feature, question, legal, off_action, context=None):
        policy_context={k:v for k,v in (context or {}).items() if k!='tracker_state_before'}
        if question=='MATCH_DECISION':
            action = self.fast_match.action(feature,legal) if self.fast_match is not None else policy.decide(feature,question,legal,off_action=off_action,context=policy_context).committed_action
        else:action=off_action
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
        if self.record_transitions:
            self.transition_buffer.append({'key':key,'kwargs':kwargs,'committed_ids':dict(result['committed_track_ids'])})
            if len(self.transition_buffer)>=64:self.flush_journal()
        self.steps_completed+=1
        if self.output is not None and time.monotonic()-self.status_stamp>=30:
            save(self.output/'runtime_status.json',{'phase':'LIVE_POLICY_ROLLOUT','last_key':list(key),
                'completed_payloads':self.steps_completed,'pid':os.getpid(),'memory_writes':state.counters.get('memory_writes',0)})
            self.status_stamp=time.monotonic()
        self.update_metadata(payload,state,result['committed_track_ids'],kwargs.get('memory_actions',{}))
        if self.after_step is not None:
            try:self.after_step(key,payload,state,result,kwargs,self.decision_rows)
            except EndEventHorizon:self.stop_requested=True
        return result

    def flush_journal(self):
        if not self.transition_buffer:return
        import torch
        path=self.output/'committed_transitions'/f'{len(self.journal_files):04d}.pth'
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(self.transition_buffer,path);self.journal_files.append(path);self.transition_buffer=[]

    def rebuild_memory_prefixes(self,selected):
        """Reconstruct clean prefixes from actual commits, without GMT calls.

        Cached actions apply only to the factual prefix. Every intervened
        current/future key still runs the live native policy in run().
        """
        from jev_counterfactual_v2 import sync_production_history_for_key
        state=self.seed_state.clone();self.metadata=copy.deepcopy(self.seed_metadata)
        wanted={tuple(r['key']):r for r in selected};captures=[]
        for path in self.journal_files:
            import torch
            for entry in torch.load(path,map_location='cpu'):
                key=tuple(entry['key']);payload=self.cache.load(*key);kwargs=entry['kwargs']
                sync_production_history_for_key(state,frame=key[1],view=key[2],view_num=2,history_limit=80)
                self.current=payload;self.prefix=state
                for track,m in self.metadata.items():
                    if track not in state.active_ids:m.setdefault('stale_since',key[1])
                    else:m.pop('stale_since',None)
                if self.native_resolver is not None:self.native_resolver.reset()
                resolution=self.engine.resolve_actions(payload,state,actions=kwargs['actions'],proposal=kwargs['proposal'])
                for record_key,record in wanted.items():
                    if record_key[:3]==key:
                        row=record_key[4]
                        if resolution['existing_track_ids'].get(row)!=record['track_id']:
                            raise AssertionError('journal replay changed selected MEMORY identity')
                        captures.append({'state':state.clone(),'metadata':copy.deepcopy(self.metadata),
                            'context':{'video_id':key[0],'frame':key[1],'view':key[2],'detection_index':row,'track_id':record['track_id'],
                                'model_image_size':list(payload['image_size']),'bbox_xyxy':payload['pred_boxes'][row].tolist()},
                            'selection':record})
                if any(t is None for t in resolution['existing_track_ids'].values()):
                    ids,recent=self.pilot.reactivation_candidates(state,bank_size=state.memory_bank_size)
                    if ids:state.stale_ids.update(ids)
                result=self.original_step(payload,state,**kwargs)
                if result['committed_track_ids']!=entry['committed_ids']:
                    raise AssertionError(f'journal native commit/RNG replay mismatch at {key}')
                if key[1]==self.seed_key[1] and key[2]!=self.seed_key[2]:
                    state.association_history.sort(key=lambda item:int(item['perception']['view']))
                self.update_metadata(payload,state,result['committed_track_ids'],kwargs.get('memory_actions',{}))
        return captures,state

    @staticmethod
    def state_fingerprint(state,metadata):
        import hashlib
        digest=hashlib.sha256()
        containers={'next_id':state.next_id,'active_ids':sorted(state.active_ids),'stale_ids':sorted(state.stale_ids),
            'possible_memory_ids':sorted(state.possible_memory_ids),'bank_order':list(state.reactivation_bank),
            'track_hits':state.track_hits,'memory_bank_size':state.memory_bank_size,
            'rng_state':state.trajectory_rng_state,'rng_calls':state.trajectory_rng_calls,
            'history':[[[item['perception']['video_id'],item['perception']['frame'],item['perception']['view']],item['assignments']] for item in state.association_history],
            'metadata':metadata}
        digest.update(json.dumps(containers,sort_keys=True).encode())
        for mapping in (state.track_embeddings,state.reactivation_bank):
            for track,vector in sorted(mapping.items()):digest.update(str(track).encode());digest.update(vector.detach().cpu().numpy().tobytes())
        for track,vectors in sorted(state.memory.items()):
            digest.update(str(track).encode())
            for vector in vectors:digest.update(vector.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def update_metadata(self,payload,state,committed,memory_actions):
        for row,track in committed.items():
            score=float(payload['detection_scores'][row]);frame=int(payload['frame']);view=int(payload['view'])
            m=self.metadata.setdefault(int(track),{'first_seen':frame,'last_seen':frame,'views':[],
                'last_confidence':score,'recent_confidences':[],'memory_quality':[]})
            m.update(last_seen=frame,last_confidence=score)
            m['views']=sorted(set(m['views'])|{view})
            m['recent_confidences']=(m['recent_confidences']+[score])[-10:]
            if memory_actions.get(row)=='WRITE_MEMORY':m['memory_quality']=(m['memory_quality']+[score])[-10:]

    def run(self, output, *, prefix=None, start_key=None, end_frame=None,
            intervention=None, before_step=None, after_step=None, metadata=None):
        pilot = self.pilot
        output = new_output(output); pilot.PILOT = output
        self.output=output;self.status_stamp=0
        self.intervention = intervention; self.before_step = before_step; self.after_step = after_step
        self.stop_requested=False; self.steps_completed=0
        self.transition_buffer=[];self.journal_files=[]
        self.metadata=copy.deepcopy(self.seed_metadata if prefix is None else metadata or {})
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
        keys=EventHorizonKeys(keys,self)
        pilot.ordered_replay_keys = lambda cache,video_id,view_num:(keys,self.seed_key,self.first_counts)
        try:
            result = pilot.run_method('jev',B2,image_lookup=self.lookup,by_key={},records={},
                feature_source_mode='runtime',parity_report={},device=self.device,**self.kwargs)
            result.update(payloads=self.steps_completed+(1 if prefix is None else 0),event_horizon_stop=self.stop_requested)
            self.flush_journal()
            result['native_match_validation']=self.native_match_validation
            save(output/'result.json',result)
            save(output/'runtime_status.json',{'phase':'COMPLETE','completed_payloads':result['payloads'],'pid':os.getpid()})
            return result
        finally:
            self.intervention = self.before_step = self.after_step = None
            pilot.append_predictions = self.original_append


def event_key(row):
    c=row['context']
    return (int(c['video_id']),int(c['frame']),int(c['view']),row['question'],int(c['detection_index']))
