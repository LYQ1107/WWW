"""Full TRAIN12 native asset construction, separate from V2/V3 and JEV.

No GT reader is imported or invoked. Every real inference observation is copied
in its native precision; only actual solver calls and actual dict commits are
observed. Existing full-audit outputs are equality references, never selectors.
"""
import argparse
import collections
import gzip
import shutil
import time
import torch
from common import *
from audit_decision_trace import records
from dev12_native_schema import INTERFACE, CommitDictionary, digest_json, tensor_digest
from gtr.geometry.decision_trace import DecisionTrace

NEW = Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
BASELINE = NEW/'full_train_audit_v1/scene12_full_v1'


def instance_digest(instance):
    return digest_json({name: tensor_digest(value.tensor if hasattr(value, 'tensor') else value)
                        for name, value in instance.get_fields().items()
                        if name in ('reid_features', 'pred_boxes', 'track_ids', 'scores', 'pred_classes')})


def state_summary(tracker, reid_digests=None):
    from gtr.modeling.meta_arch.gtr_rcnn import poss_ids, old_ids, old_reids
    return dict(counts={str(k): int(v) for k,v in sorted(tracker.counts.items())},
                reid_digests={str(k): v for k,v in sorted((reid_digests if reid_digests is not None else
                    {k: instance_digest(v) for k,v in tracker.reids.items()}).items())},
                possible_bank_ids=sorted(map(int, poss_ids.poss_ids)), old_ids=sorted(map(int, old_ids.old_ids)),
                bank_instances=[instance_digest(v) for v in old_reids.old_reids])


class AssetTrace(DecisionTrace):
    def __init__(self, model, out, metadata):
        super().__init__(model, out)
        self.out=out; self.metadata=metadata; self.observations={}; self.variants={}; self.calls=[]
        self.call_shards=[]; self.query_count=0; self.primary_count=0; self.active=[]; self.next_call=0
        self.reid_digests={}; self.commits=gzip.open(out/'native_commit_trace.jsonl.gz','xt')
        self.query_stream=gzip.open(out/'native_queries.jsonl.gz','xt')
        self.commit_count=0; self.previous_commit=None; self.completed_images=set(); self.pending_public={}
        self.bank_support={}; self.tracker=None

    def write(self, stream, record):
        super().write(stream, record)
        if stream is self.detectors:
            self.pending_public[tuple(record['source'])]=record['detection']
        elif self.active:
            self.active[-1]['decision']=record

    def inference(self,*args,**kwargs):
        result=super().inference(*args,**kwargs)
        for inst in result:
            if inst.reid_features.shape[-1]!=1152:raise ValueError('native dimension changed')
            for i, source in enumerate(inst.decision_trace_source.cpu().tolist()):
                key=tuple(source)
                if key in self.observations:raise ValueError('duplicate physical source')
                row=self.metadata[key[0]]; public=self.pending_public.pop(key)
                feature=inst.reid_features[i].detach().cpu().clone()
                if feature.dtype!=torch.float32:raise ValueError('native precision differs from declared FP32')
                digest=tensor_digest(feature)
                self.observations[key]=dict(source=list(key),image_id=key[0],detection_row=key[1],frame=row['frame_id'],
                    camera=row['view_id'],width=row['width'],height=row['height'],path=row['file_name'],
                    bbox_xywh=public['bbox'],detection_score=public['score'],native_box=inst.pred_boxes.tensor[i].detach().cpu().clone(),
                    original_dtype=str(feature.dtype),feature=feature,feature_sha256=digest)
        return result

    def feature_key(self, source, feature):
        key=tuple(source); value=feature.detach().cpu().clone(); digest=tensor_digest(value)
        if digest==self.observations[key]['feature_sha256']:return dict(source=list(key),variant=None)
        # Native memory_bank can mutate a descriptor sharing its original source
        # tag. Preserve that real vector as a version, never overwrite the raw
        # detector observation or pretend an average is a single person's ROI.
        variant=digest_json([list(key),digest])
        if variant not in self.variants:self.variants[variant]=dict(source=list(key),feature=value,feature_sha256=digest,
            semantics='actual native descriptor version; not an unmodified detector observation')
        return dict(source=list(key),variant=variant)

    def commit(self, identity, value):
        self.reid_digests[identity]=instance_digest(value)
        summary=state_summary(self.tracker,self.reid_digests)
        source=value.decision_trace_source[-1].cpu().tolist()
        row=dict(index=self.commit_count,source=source,track=identity,call_id=self.active[-1]['call_id'] if self.active else None,
            reid_digest=self.reid_digests[identity],state=summary,state_sha256=digest_json(summary),previous_commit_sha256=self.previous_commit)
        row['commit_sha256']=digest_json(row)
        self.commits.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        self.previous_commit=row['commit_sha256']; self.commit_count+=1
        if not self.active:self.completed_images.add(source[0])

    def method(self,name,args,kwargs):
        offset=1 if name=='run_global_tracker_plus' else 0
        instances=args[offset]; k=args[offset+3]; query=instances[k]
        refs=[inst for i,inst in enumerate(instances) if i!=k]
        sources=query.decision_trace_source.cpu().tolist(); ref_sources=[s for inst in refs for s in inst.decision_trace_source.cpu().tolist()]
        ref_tracks=[int(t) for inst in refs for t in inst.track_ids.cpu().tolist()]
        tracks=sorted(set(ref_tracks)); reference_counts=collections.Counter(ref_tracks)
        call=dict(call_id=self.next_call,stage=name,primary_index=self.primary_count if name!='run_memory_tracker' else None,
            queries=sources,references=[dict(source=s,track=t) for s,t in zip(ref_sources,ref_tracks)],track_ids=tracks,
            reference_block_sizes=[len(inst) for inst in refs],actual_candidate_support_lengths=[reference_counts[t] for t in tracks],
            observed_track_lengths={str(t):int(self.tracker.counts[t]) for t in tracks},
            query_native_context_indices=list(range(sum(len(inst) for inst in instances[:k]),sum(len(inst) for inst in instances[:k+1]))),
            query_feature_keys=[self.feature_key(s,f) for s,f in zip(sources,query.reid_features)],
            reference_feature_keys=[self.feature_key(s,f) for inst in refs for s,f in zip(inst.decision_trace_source.cpu().tolist(),inst.reid_features)],
            query_native_boxes=query.pred_boxes.tensor.detach().cpu().clone(),
            reference_native_boxes=torch.cat([inst.pred_boxes.tensor for inst in refs]).detach().cpu().clone(),
            raw_logits=args[offset+1][-1].detach().cpu().clone(),
            threshold_base=float(self.model.thred_bank if name=='run_memory_tracker' else self.model.overlap_thresh),
            not_mult_thresh=bool(self.model.not_mult_thresh),with_bank=bool(self.model.with_bank),test_len=int(self.model.test_len),
            id_count_before=int(args[offset+4]) if name!='run_memory_tracker' else None,
            state_before=state_summary(self.tracker,self.reid_digests),
            completed_camera_images_before_call=sorted(self.completed_images))
        call['thresholds']=torch.tensor(call['threshold_base'],dtype=torch.float32)*torch.tensor([1 if self.model.not_mult_thresh else reference_counts[t] for t in tracks],dtype=torch.float32)
        call['activation_block_sizes']=([len(ref_sources)] if name!='run_global_tracker_plus' else call['reference_block_sizes'])
        if name=='run_memory_tracker':call['bank_descriptor_support']={str(t):self.bank_support.get(t) for t in tracks}
        self.next_call+=1
        if name!='run_memory_tracker':self.primary_count+=1
        self.active.append(call)
        try:
            result=super().method(name,args,kwargs)
            call['state_after']=state_summary(self.tracker,self.reid_digests)
            call['id_count_after']=int(result[1]) if name!='run_memory_tracker' else None
            self.calls.append(call)
            if name!='run_memory_tracker':
                for row,source in enumerate(sources):
                    self.query_stream.write(json.dumps(dict(key=[12,*source],call_id=call['call_id'],primary_index=call['primary_index'],query_row=row,
                        primary_accepted_track=call['decision']['accepted_track_ids'][row],final_track=call['decision']['final_track_ids'][row]),separators=(',',':'))+'\n')
                    self.query_count+=1
                if sources:self.completed_images.add(sources[0][0])
            return result
        finally:self.active.pop()

    def solve(self,cost,*args,**kwargs):
        if self.active:
            self.active[-1]['track_scores']=-cost.detach().cpu().clone()
            tensor_digest(cost)
        return super().solve(cost,*args,**kwargs)

    def install(self):
        super().install()
        self.original_activate=self.model.roi_heads._activate_asso
        def activated(*args,**kwargs):
            result=self.original_activate(*args,**kwargs)
            if self.active:self.active[-1]['single_reference_probabilities']=torch.cat(result,1).detach().cpu().clone()
            return result
        self.model.roi_heads._activate_asso=activated
        self.original_memory=self.model.memory_bank
        def memory(counts,reids,unique_ids,older,no_match):
            if older and len(no_match[0]):
                from gtr.modeling.meta_arch.gtr_rcnn import poss_ids
                active=set(map(int,unique_ids.cpu().tolist()))
                for identity in poss_ids.poss_ids.copy():
                    if identity in active:continue
                    history=reids[identity]
                    support=[self.feature_key(history.decision_trace_source[-1-i].cpu().tolist(),history.reid_features[-1-i]) for i in range(self.model.bank_size)]
                    self.bank_support[int(identity)]=dict(operation='ordered sum then divide by bank_size; native code',sources=support,denominator=int(self.model.bank_size))
            result=self.original_memory(counts,reids,unique_ids,older,no_match)
            # The original Bank may update a shared descriptor in-place.
            self.reid_digests.update({int(t):instance_digest(v) for t,v in reids.items()})
            return result
        self.model.memory_bank=memory

    def flush(self):
        if not self.calls:return
        path=self.out/'native_calls'/f'calls_{len(self.call_shards):04d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(sorted(self.calls,key=lambda c:c['call_id']),path)
        self.call_shards.append(str(path.relative_to(self.out)));self.calls=[]

    def close(self):
        self.flush();self.commits.close();self.query_stream.close()
        if hasattr(self,'original_activate'):self.model.roi_heads._activate_asso=self.original_activate
        if hasattr(self,'original_memory'):self.model.memory_bank=self.original_memory
        super().close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',type=int,choices=[40,1200],required=True);p.add_argument('--export',choices=['on','off'],required=True)
    a=p.parse_args();scope=json.loads(a.scope.read_text())
    if scope['interface_version']!=INTERFACE or scope['status']!='FROZEN_GT_INDEPENDENT_SCOPE':raise ValueError('scope not frozen')
    if a.groups==1200:
        gate=json.loads((NEW/'dev12_native_precheck_v1/gate.json').read_text())
        if gate['status']!='PASS' or gate['scope_sha256']!=sha256(a.scope):raise ValueError('real40 gate required')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);started=time.perf_counter();wall_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    prov=provenance()
    for key in ['checkpoint_sha256','config_sha256','annotation_sha256','seed']:
        if prov[key]!=scope['frozen_provenance'][key]:raise ValueError('source mismatch '+key)
    for path,digest in scope['bound_source_hashes'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('frozen source changed '+path)
    if scope['capture_source_sha256']!=sha256(Path(__file__)):raise ValueError('capture code changed after scope lock')
    data=train_data();images=video_records(data,12,a.groups);by_image={i['image_id']:i for i in images};groups=collections.defaultdict(list)
    for image in images:groups[image['frame_id']].append(image)
    if list(sorted(groups))!=list(range(1,a.groups+1)):raise ValueError('continuous range changed')
    if [r['image_id'] for rows in groups.values() for r in rows]!=scope['physical_input_image_ids'][:a.groups*2]:raise ValueError('physical camera order changed')
    # All current frames are bound by bytes before entering inference. GT files
    # are absent from this list, and never opened by this exporter.
    for image in images:
        if sha256(image['file_name'])!=scope['input_image_sha256'][str(image['image_id'])]:raise ValueError('image bytes changed')
    if shutil.disk_usage(a.output).free<scope['minimum_free_bytes']:raise ValueError('free disk reserve violated')
    cfg,model=load_gmt()
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    seed_all(20261014);tracker=CausalGMT(model);trace=None;predictions=[];states=[]
    initial=dict(status='RUNNING',interface_version=INTERFACE,scope_sha256=sha256(a.scope),scope=str(a.scope),export_enabled=a.export=='on',
        groups_requested=a.groups,provenance=prov,started_utc=wall_started,GT_read=False,selection='all actual primary queries in continuous groups1..1200; no GT selection')
    save_json(a.output/'manifest.json',initial)
    if a.export=='on':
        trace=AssetTrace(model,a.output,by_image);trace.tracker=tracker
        tracker.reids=CommitDictionary(trace.commit);trace.install()
    try:
        for frame,group in sorted(groups.items()):
            if time.perf_counter()-started>scope['maximum_single_process_seconds']:raise ValueError('independent GPU budget exhausted')
            if trace:trace.completed_images.clear()
            output=tracker.step(map_records(cfg,group))
            for record,result in zip(group,output):predictions.extend(custom_instances_to_coco_json(result['instances'].to('cpu'),record['image_id']))
            if a.groups==40:
                summary=state_summary(tracker);summary['id_count']=tracker.id_count
                summary['gallery_track_ids']=[x.track_ids.cpu().tolist() for x in tracker.instances]
                states.append(dict(frame=frame,state_sha256=digest_json(summary)))
            if trace and frame%100==0:trace.flush()
            if frame%100==0:
                size=sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())
                if size>scope['maximum_output_bytes']:raise ValueError('independent storage cap exceeded')
                print(json.dumps(dict(groups=frame,queries=trace.query_count if trace else None,seconds=time.perf_counter()-started,bytes=size)),flush=True)
    finally:
        if trace:trace.close()
    torch.cuda.synchronize();save_json(a.output/'predictions.json',predictions)
    if states:save_json(a.output/'group_state_fingerprints.json',states)
    bm=json.loads((BASELINE/'manifest.json').read_text());expected=[r for r in json.loads((BASELINE/'predictions.json').read_text()) if r['image_id'] in by_image]
    if predictions!=expected:raise ValueError('native predictions changed; no release')
    if trace:
        expected_decisions=iter(records(BASELINE/'actual_decisions.jsonl.gz'));n=0
        for record in records(a.output/'actual_decisions.jsonl.gz'):
            if next(expected_decisions)!=record:raise ValueError('actual native solver/commit changed; no release')
            n+=1
        try:
            if a.groups==1200:next(expected_decisions);raise ValueError('missing native decision')
        except StopIteration:pass
        obsdir=a.output/'native_observations';obsdir.mkdir()
        rows=sorted(trace.observations.items())
        for index,start in enumerate(range(0,len(rows),2048)):torch.save(dict(rows[start:start+2048]),obsdir/f'observations_{index:04d}.pt')
        torch.save(trace.variants,obsdir/'descriptor_variants.pt')
        initial.update(native_queries=trace.query_count,primary_calls=trace.primary_count,total_calls=trace.next_call,native_commits=trace.commit_count,
            native_observations=len(rows),descriptor_versions=len(trace.variants),call_shards=trace.call_shards,actual_decisions_exact=n,
            feature_dtype='torch.float32',native_feature_dimension=1152,lossy_compression=False,GT_label_files_present=False)
    seconds=time.perf_counter()-started
    initial.update(status='CAPTURE_COMPLETE_PARITY_PASS_UNLABELED',groups=len(groups),images=len(images),predictions_equal_sealed_native=True,
        full_prediction_file_bytes_exact=sha256(a.output/'predictions.json')==bm['predictions_sha256'] if a.groups==1200 else 'PREFIX40',
        process_seconds=seconds,gpu_wall_hours=seconds/3600,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),speed_eligible=False,
        artifacts={str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file() and f.name!='manifest.json'})
    save_json(a.output/'manifest.json',initial);print(json.dumps({k:v for k,v in initial.items() if k not in ('provenance','artifacts')}),flush=True)


if __name__=='__main__':main()
