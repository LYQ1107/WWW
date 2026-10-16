"""Bounded metadata ledger of native commits, without feature copies or GT.

A contiguous (camera, native-ID) chain is a recorded fragment hypothesis, not a
certified physical person. Ledger relations alone never implement native SPLIT.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from .types import IdentityFragment,SourceKey


class FragmentLedger:
    def __init__(self,max_history=16,max_fragments_per_id=32):
        if max_history<1 or max_fragments_per_id<1:raise ValueError('Positive finite ledger limits required')
        self.max_history=max_history;self.max_fragments=max_fragments_per_id;self.reset()

    def reset(self):
        self.video=None;self.latest_frame=0;self.fragments=OrderedDict();self.latest={}
        self.total_observations=0;self.journal_sha256='0'*64

    def register_commit(self,source,native_global_id):
        if not isinstance(source,SourceKey):raise TypeError('Whitelisted source metadata required')
        if native_global_id<1:raise ValueError('Only actual final native commits can enter the ledger')
        if self.video is None:self.video=source.video
        if source.video!=self.video:raise ValueError('Explicit video reset required')
        if source.frame<self.latest_frame:raise ValueError('Future-to-past insertion')
        slot=(source.camera,native_global_id);previous=self.latest.get(slot)
        fragment=self.fragments.get(previous)
        if fragment is not None and fragment.latest_frame==source.frame:
            raise ValueError('Two current detections occupy one camera/global ID')
        if fragment is None or fragment.latest_frame!=source.frame-1:
            fragment=IdentityFragment(source,native_global_id,native_global_id,source.camera,source.frame,source.frame)
            same=[key for key,value in self.fragments.items() if value.current_global_id==native_global_id]
            if len(same)>=self.max_fragments:
                evict=same[0];old=self.fragments[evict]
                if old.published_through<old.latest_frame:raise RuntimeError('Cannot evict unpublished revision evidence')
                del self.fragments[evict]
                for key,value in list(self.latest.items()):
                    if value==evict:del self.latest[key]
            self.fragments[source]=fragment;self.latest[slot]=source
        fragment.latest_frame=source.frame;fragment.observations_seen+=1
        fragment.observation_source_keys.append(source)
        fragment.observation_source_keys=fragment.observation_source_keys[-self.max_history:]
        self.total_observations+=1;self.latest_frame=source.frame
        event=dict(source=asdict(source),native_id=native_global_id,fragment=asdict(fragment.fragment_key),
                   relation_version=fragment.relation_version,previous=self.journal_sha256)
        self.journal_sha256=hashlib.sha256(json.dumps(event,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return fragment.fragment_key

    def mark_published(self,frame):
        if frame>self.latest_frame:raise ValueError('Cannot publish future observations')
        for fragment in self.fragments.values():
            fragment.published_through=max(fragment.published_through,min(frame,fragment.latest_frame))

    def validate_invariants(self):
        per_id={}
        for key,f in self.fragments.items():
            if key!=f.fragment_key or not f.observation_source_keys or len(f.observation_source_keys)>self.max_history:
                raise RuntimeError('Ledger source/retention mismatch')
            if f.latest_frame>self.latest_frame or any(s.video!=self.video or s.frame>f.latest_frame or s.camera!=f.camera for s in f.observation_source_keys):
                raise RuntimeError('Ledger source chronology mismatch')
            if f.current_global_id!=f.original_global_id or f.relation_version!=0:
                raise RuntimeError('Native revision is not qualified for this observe-only ledger')
            per_id[f.current_global_id]=per_id.get(f.current_global_id,0)+1
        if any(n>self.max_fragments for n in per_id.values()):raise RuntimeError('Unbounded per-ID fragments')
        if any(key not in self.fragments for key in self.latest.values()):raise RuntimeError('Stale fragment index')
        return True

    def state_digest(self):
        payload=dict(video=self.video,latest_frame=self.latest_frame,total=self.total_observations,
                     journal=self.journal_sha256,fragments=[asdict(f) for f in self.fragments.values()])
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
