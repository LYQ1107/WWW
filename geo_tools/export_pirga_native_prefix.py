"""Fresh native fit-scene prefix resource probe. Requires explicit budget permit.

No GT read, label recomputation, event selection or training. A passive observer
streams full native rows/columns into the indexed store. Export OFF and ON are
separate cold starts and must later pass actual output/state/decision parity.
"""
import argparse
import collections
import json
from pathlib import Path
import shutil
import time


def validate_permit(path,digest,scene,groups):
    from pirga_indexed_store import sha
    if sha(path)!=digest:raise ValueError('Unpinned resource permit')
    p=json.loads(Path(path).read_text())
    if (p.get('status')!='AUTHORIZED' or not p.get('authorization_source') or p['scope']!='PIRGA_FIT_PREFIX_OFF_ON'
        or scene!=4 or groups!=40 or p['scene']!=4 or p['groups']!=40 or p['maximum_process_seconds']>360
        or not 0<p['maximum_total_gpu_wall_hours']<=.25 or not 0<p['maximum_process_seconds']<=360 or p['old_ledgers_reset'] is not False):raise ValueError('Scope/budget not authorized')
    root=Path(__file__).resolve().parents[1]
    expected={'geo_tools/export_pirga_native_prefix.py','geo_tools/run_pirga_prefix_pair.py','geo_tools/pirga_native_trace.py','geo_tools/pirga_trace_primitives.py','geo_tools/pirga_indexed_store.py'}
    if set(p.get('bound_capture_source_sha256',{}))!=expected:raise ValueError('Capture source set is not frozen')
    for name,digest in p['bound_capture_source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Capture source changed '+name)
    return p


def main():
    p=argparse.ArgumentParser();p.add_argument('--permit',type=Path,required=True);p.add_argument('--permit-sha',required=True)
    p.add_argument('--scene',type=int,choices=[4],default=4);p.add_argument('--groups',type=int,choices=[40],default=40)
    p.add_argument('--export',choices=['off','on'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();permit=validate_permit(a.permit,a.permit_sha,a.scene,a.groups)
    started=time.monotonic()
    from common import (sha256,save_json,train_data,video_records,seed_all,load_gmt,map_records,provenance,ROOT,CHECKPOINT)
    import torch
    from pirga_native_trace import AssetTrace,state_summary
    from pirga_trace_primitives import CommitDictionary
    from pirga_indexed_store import StoreWriter,sparse_matrix
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    if a.out.exists():raise FileExistsError(a.out)
    if shutil.disk_usage(a.out.parent).free<30*(1<<30)+permit['maximum_output_bytes']:raise ValueError('30GiB reserve plus output cap required')
    if sha256(CHECKPOINT)!='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8':raise ValueError('Original Stage2 changed')
    for name,digest in permit['bound_native_source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Native source changed '+name)
    base=Path(permit['sealed_native_audit']);bm=json.loads((base/'manifest.json').read_text())
    if sha256(base/'manifest.json')!=permit['sealed_native_audit_manifest_sha256']:raise ValueError('Source audit changed')
    if sha256(base/'predictions.json')!=bm['predictions_sha256']:raise ValueError('Source predictions changed')
    # Existing train annotation is accessed only through image metadata, as in
    # original GMT. No GT matcher, labels or offline event file enters inference.
    data=train_data();images=video_records(data,a.scene,a.groups);meta={r['image_id']:r for r in images}
    if len(images)!=80 or set(r['frame_id'] for r in images)!=set(range(1,41)):raise ValueError('Continuous scope mismatch')
    groups=collections.defaultdict(list)
    for r in images:groups[r['frame_id']].append(r)
    a.out.mkdir(parents=True)
    scope=dict(status='FROZEN_CONTINUOUS_PREFIX_BEFORE_INFERENCE',scene=a.scene,groups=[1,40],
               images=[dict(image_id=r['image_id'],frame=r['frame_id'],camera=r['view_id'],sha256=sha256(r['file_name'])) for r in images])
    save_json(a.out/'input_scope.json',scope)
    cfg,model=load_gmt();seed_all(20261014);tracker=CausalGMT(model)
    writer=StoreWriter(a.out/'indexed_inputs') if a.export=='on' else None
    class IndexedTrace(AssetTrace):
        def fid(self,spec):
            key=tuple(spec['source']);o=self.observations[key];v=o if spec['variant'] is None else self.variants[spec['variant']]
            return writer.add_feature(a.scene,list(key),o['frame'],o['camera'],v['feature'].numpy(),spec['variant'])
        def flush(self):
            for c in sorted(self.calls,key=lambda c:c['call_id']):
                shape,entries=sparse_matrix(c['track_scores'].numpy())
                writer.add_call(dict(scene=a.scene,call_id=c['call_id'],stage=c['stage'],query_keys=[[a.scene,*q] for q in c['queries']],
                    query_features=[self.fid(s) for s in c['query_feature_keys']],reference_features=[self.fid(s) for s in c['reference_feature_keys']],
                    reference_tracks=[r['track'] for r in c['references']],candidate_ids=c['track_ids'],reference_block_sizes=c['reference_block_sizes'],
                    matrix_shape=shape,matrix_entries=entries,thresholds=c['thresholds'].tolist(),accepted_ids=c['decision']['accepted_track_ids'],
                    completed_image_ids=c['completed_camera_images_before_call'],
            observed_track_lengths=c['observed_track_lengths'],candidate_support_counts=c['actual_candidate_support_lengths']))
            self.calls=[]
    trace=None;predictions=[];states=[]
    prov=provenance()
    for name in ('checkpoint_sha256','config_sha256','annotation_sha256','seed'):
        if prov[name]!=bm['provenance'][name]:raise ValueError('Frozen source provenance differs '+name)
    initial=dict(status='RUNNING',scene=a.scene,groups_requested=a.groups,export=a.export,GT_read=False,teacher_calls=0,training=False,
                 resource_permit_sha256=a.permit_sha,provenance=prov,input_scope_sha256=sha256(a.out/'input_scope.json'))
    save_json(a.out/'manifest.json',initial)
    if writer:
        trace=IndexedTrace(model,a.out,meta,a.scene);trace.tracker=tracker;tracker.reids=CommitDictionary(trace.commit);trace.install()
    try:
        for frame,group in sorted(groups.items()):
            if time.monotonic()-started>permit['maximum_process_seconds']-30:raise RuntimeError('Resource limit reached')
            if trace:trace.completed_images.clear()
            output=tracker.step(map_records(cfg,group))
            for record,result in zip(group,output):predictions.extend(custom_instances_to_coco_json(result['instances'].to('cpu'),record['image_id']))
            state=state_summary(tracker);state['id_count']=tracker.id_count;state['gallery_track_ids']=[x.track_ids.cpu().tolist() for x in tracker.instances]
            from pirga_trace_primitives import digest_json
            states.append(dict(frame=frame,state_sha256=digest_json(state)))
            if trace:trace.flush()
            if frame%10==0:
                used=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
                if used>permit['maximum_output_bytes'] or shutil.disk_usage(a.out).free<30*(1<<30):raise RuntimeError('Storage limit')
                print(json.dumps(dict(frame=frame,seconds=time.monotonic()-started,bytes=used,queries=trace.query_count if trace else None)),flush=True)
    finally:
        if trace:trace.close()
    torch.cuda.synchronize();save_json(a.out/'predictions.json',predictions);save_json(a.out/'group_state_fingerprints.json',states)
    original=json.loads((base/'predictions.json').read_text());expected=[r for r in original if r['image_id'] in meta]
    if predictions!=expected:raise ValueError('Passive exporter altered source predictions')
    if trace:
        from audit_decision_trace import records
        actual=list(records(a.out/'actual_decisions.jsonl.gz'));original=list(records(base/'actual_decisions.jsonl.gz'))
        if actual!=original[:len(actual)]:raise ValueError('Passive exporter altered solver/threshold/commit')
        for key,o in trace.observations.items():writer.add_feature(a.scene,list(key),o['frame'],o['camera'],o['feature'].numpy())
        store=writer.close(dict(permit=a.permit_sha,native_source=permit['bound_native_source_sha256']),trace.query_count)
        initial.update(queries=trace.query_count,calls=trace.next_call,observations=len(trace.observations),variants=len(trace.variants),indexed_store=store)
    initial.update(status='PASS_PREFIX_NATIVE_EXPORT_SOURCE_PARITY_PENDING_OFF_ON_GATE',groups=40,predictions_exact_source_prefix=True,
                   process_body_wall_seconds=time.monotonic()-started,fresh_image_inference=True,speed_eligible=False,
                   artifacts={str(f.relative_to(a.out)):sha256(f) for f in a.out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    save_json(a.out/'manifest.json',initial);print(json.dumps({k:v for k,v in initial.items() if k not in ('provenance','artifacts','indexed_store')}),flush=True)

if __name__=='__main__':main()
