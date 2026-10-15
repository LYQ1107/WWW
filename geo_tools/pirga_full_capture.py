"""Passive full native input capture, bounded indexed storage, no GT semantics.

No tracker, GTA, thresholds, Gallery or Bank algorithm is replaced. DecisionTrace
calls the original functions and retains their returned objects. Unlike the
small state-parity probe, the full stream records compact commit links instead
of repeating the entire Gallery in every commit. Actual descriptor values are
snapshotted from the real query/reference tensors, including native versions.
"""
from collections import Counter
import gzip
import hashlib
import json
import numpy as np
import torch
from gtr.geometry.decision_trace import DecisionTrace
from pirga_indexed_store import sparse_matrix
from pirga_trace_primitives import digest_json


class FullNativeTrace(DecisionTrace):
    def __init__(self,model,out,metadata,scene,writer):
        super().__init__(model,out);self.out=out;self.metadata=metadata;self.scene=scene;self.writer=writer
        self.observations={};self.versions={};self.active=[];self.pending=[];self.next_call=0;self.query_count=0;self.commit_count=0
        self.completed_images=set();self.previous_commit=None;self.tracker=None
        self.commits=gzip.open(out/'compact_commit_trace.jsonl.gz','xt')
        self.query_stream=gzip.open(out/'native_queries.jsonl.gz','xt')

    def write(self,stream,record):
        super().write(stream,record)
        if stream is self.decisions and self.active:self.active[-1]['decision']=record

    @staticmethod
    def feature_digest(value):
        x=np.asarray(value,dtype=np.float32)
        return hashlib.sha256(x.dtype.str.encode()+x.tobytes()).hexdigest()

    def inference(self,*args,**kwargs):
        result=super().inference(*args,**kwargs)
        for inst in result:
            if inst.reid_features.shape[-1]!=1152 or inst.reid_features.dtype!=torch.float32:raise ValueError('Native1152 contract changed')
            features=inst.reid_features.detach().cpu().contiguous().numpy()
            for source,value in zip(inst.decision_trace_source.cpu().tolist(),features):
                key=tuple(source);meta=self.metadata[key[0]]
                if key in self.observations:raise ValueError('Repeated physical observation')
                fid=self.writer.add_feature(self.scene,source,meta['frame_id'],meta['view_id'],value)
                self.observations[key]=dict(fid=fid,raw_digest=self.feature_digest(value))
        return result

    def feature_ids(self,instance):
        if instance.reid_features.dtype!=torch.float32:raise ValueError('Actual native descriptor dtype differs from FP32 source')
        features=instance.reid_features.detach().cpu().contiguous().numpy();result=[]
        for source,value in zip(instance.decision_trace_source.cpu().tolist(),features):
            key=tuple(source);raw=self.observations[key];digest=self.feature_digest(value)
            if digest==raw['raw_digest']:fid=raw['fid']
            else:
                variant=digest_json([source,digest]);vk=(key,variant)
                if vk not in self.versions:
                    meta=self.metadata[key[0]]
                    self.versions[vk]=self.writer.add_feature(self.scene,source,meta['frame_id'],meta['view_id'],value,variant)
                fid=self.versions[vk]
            result.append(fid)
        return result

    def commit(self,identity,value):
        source=value.decision_trace_source[-1].cpu().tolist()
        row=dict(index=self.commit_count,source=source,track=identity,call_id=self.active[-1]['call_id'] if self.active else None,
                 previous_commit_sha256=self.previous_commit,semantics='actual native dict commit link; no counterfactual state')
        row['commit_sha256']=digest_json(row);self.previous_commit=row['commit_sha256'];self.commit_count+=1
        self.commits.write(json.dumps(row,separators=(',',':'))+'\n')
        if not self.active:self.completed_images.add(source[0])

    def method(self,name,args,kwargs):
        offset=1 if name=='run_global_tracker_plus' else 0;instances=args[offset];k=args[offset+3];q=instances[k]
        refs=[x for i,x in enumerate(instances) if i!=k];sources=q.decision_trace_source.cpu().tolist()
        tracks=[int(t) for x in refs for t in x.track_ids.cpu().tolist()];ids=sorted(set(tracks));support=Counter(tracks)
        base=self.model.thred_bank if name=='run_memory_tracker' else self.model.overlap_thresh
        threshold=(torch.tensor(base,dtype=torch.float32)*torch.tensor([1 if self.model.not_mult_thresh else support[t] for t in ids],dtype=torch.float32)).tolist()
        c=dict(scene=self.scene,call_id=self.next_call,stage=name,query_keys=[[self.scene,*s] for s in sources],query_features=self.feature_ids(q),
               reference_features=[fid for x in refs for fid in self.feature_ids(x)],reference_tracks=tracks,candidate_ids=ids,
               reference_block_sizes=[len(x) for x in refs],thresholds=threshold,completed_image_ids=sorted(self.completed_images),
               observed_track_lengths={str(t):int(self.tracker.counts[t]) for t in ids},candidate_support_counts=[support[t] for t in ids])
        self.next_call+=1;self.active.append(c)
        try:
            result=super().method(name,args,kwargs);decision=c.pop('decision');c['accepted_ids']=decision['accepted_track_ids']
            if name!='run_memory_tracker':
                for row,source in enumerate(sources):
                    self.query_stream.write(json.dumps(dict(key=[self.scene,*source],call_id=c['call_id'],query_row=row,
                        primary_accepted_track=decision['accepted_track_ids'][row],final_track=decision['final_track_ids'][row]),separators=(',',':'))+'\n')
                    self.query_count+=1
                if sources:self.completed_images.add(sources[0][0])
            self.pending.append(c);return result
        finally:self.active.pop()

    def solve(self,cost,*args,**kwargs):
        if self.active:
            shape,entries=sparse_matrix((-cost.detach().cpu()).numpy());self.active[-1]['matrix_shape']=shape;self.active[-1]['matrix_entries']=entries
        return super().solve(cost,*args,**kwargs)

    def flush(self):
        for c in sorted(self.pending,key=lambda c:c['call_id']):self.writer.add_call(c)
        self.pending=[]

    def close(self):
        self.flush();self.commits.close();self.query_stream.close();super().close()
