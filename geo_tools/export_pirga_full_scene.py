"""One frozen complete fit scene, or the registered real40 collector precheck."""
import argparse
from collections import defaultdict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--permit',type=Path,required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--scene',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prefix40',action='store_true');a=p.parse_args()
    started=time.monotonic()
    from pirga_indexed_store import sha,StoreWriter
    if sha(a.permit)!=a.permit_sha:raise ValueError('Unpinned export allocation')
    permit=json.loads(a.permit.read_text())
    if permit['status']!='ALLOCATED_WITHIN_EXISTING_CAP' or a.scene not in permit['fit_scenes']:raise ValueError('Unregistered export scope')
    sp=Path(permit['scope_path'])
    if sha(sp)!=permit['scope_sha256']:raise ValueError('Unpinned complete input scope')
    scope=json.loads(sp.read_text());scene=next(s for s in scope['scenes'] if s['scene']==a.scene)
    if scope['fit_scenes']!=[4,5,6,7,8,9,13,17,20,21,24] or scope['expected_total_queries']!=154911:raise ValueError('Unauthorized fit/development mixture')
    from common import ROOT,sha256,save_json,load_gmt,seed_all,map_records,provenance,CHECKPOINT
    for f,digest in scope['bound_capture_and_native_source_sha256'].items():
        if sha256(ROOT/f)!=digest:raise ValueError('Capture/native source changed '+f)
    if a.out.exists():raise FileExistsError(a.out)
    if shutil.disk_usage(a.out.parent).free<30*(1<<30)+permit['maximum_total_output_bytes']:raise ValueError('Reserve plus stage output cap required')
    mp=Path(scene['original_audit_manifest']);bm=json.loads(mp.read_text());base=mp.parent
    for path,digest in [(mp,scene['original_audit_manifest_sha256']),(base/'predictions.json',scene['original_predictions_sha256']),(base/'actual_decisions.jsonl.gz',scene['original_decisions_sha256'])]:
        if sha256(path)!=digest:raise ValueError('Sealed native reference changed')
    if sha256(CHECKPOINT)!=scope['original_checkpoint_sha256']:raise ValueError('Frozen Stage2 changed')
    limit=40 if a.prefix40 else scene['groups'];images=[r for r in scene['images'] if r['frame_id']<=limit]
    if set(r['frame_id'] for r in images)!=set(range(1,limit+1)) or len(images)!=2*limit:raise ValueError('Continuous whole-scene scope changed')
    for row in images:
        if sha256(row['file_name'])!=scene['image_sha256'][str(row['image_id'])]:raise ValueError('Input image bytes changed')
    import torch
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    from pirga_full_capture import FullNativeTrace
    from pirga_trace_primitives import CommitDictionary,digest_json
    from pirga_native_trace import state_summary
    from audit_pirga_fulltrain import read_rows
    a.out.mkdir(parents=True);cfg,model=load_gmt();seed_all(20261014);tracker=CausalGMT(model)
    writer=StoreWriter(a.out/'indexed_inputs');metadata={r['image_id']:r for r in images}
    trace=FullNativeTrace(model,a.out,metadata,a.scene,writer);trace.tracker=tracker;tracker.reids=CommitDictionary(trace.commit);trace.install()
    initial=dict(status='RUNNING',scene=a.scene,groups_requested=limit,prefix40=a.prefix40,scope_sha256=permit['scope_sha256'],permit_sha256=a.permit_sha,
                 GT_semantics_loaded=False,raw_GT_recomputed=False,teacher_calls=0,training=False,provenance=provenance(),state_recording='compact actual commit hash chain; real40 group state fingerprints separately')
    for k in ('checkpoint_sha256','config_sha256','annotation_sha256','seed'):
        if initial['provenance'][k]!=bm['provenance'][k]:raise ValueError('Frozen inference provenance changed '+k)
    save_json(a.out/'manifest.json',initial);groups=defaultdict(list)
    for row in images:groups[row['frame_id']].append(row)
    predictions=[];states=[];last_frame=0
    try:
        for frame,group in sorted(groups.items()):
            if time.monotonic()-started>permit['maximum_single_process_seconds']-15:raise RuntimeError('Single-process cap reached')
            trace.completed_images.clear();output=tracker.step(map_records(cfg,group));trace.flush();last_frame=frame
            for row,result in zip(group,output):predictions.extend(custom_instances_to_coco_json(result['instances'].to('cpu'),row['image_id']))
            if frame<=40:
                summary=state_summary(tracker);summary['id_count']=tracker.id_count;summary['gallery_track_ids']=[x.track_ids.cpu().tolist() for x in tracker.instances]
                states.append(dict(frame=frame,state_sha256=digest_json(summary)))
            if frame%20==0:
                if shutil.disk_usage(a.out).free<30*(1<<30):raise RuntimeError('Disk reserve violated')
                used=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
                if used>permit['maximum_total_output_bytes']:raise RuntimeError('Stage storage cap violated by scene')
                print(json.dumps(dict(scene=a.scene,groups=frame,queries=trace.query_count,calls=trace.next_call,bytes=used,seconds=time.monotonic()-started)),flush=True)
    except BaseException as e:
        initial.update(status='FAILED_NO_ADMISSION',reason=repr(e),completed_groups=last_frame,queries=trace.query_count);save_json(a.out/'manifest.json',initial);raise
    finally:trace.close()
    torch.cuda.synchronize();save_json(a.out/'predictions.json',predictions);save_json(a.out/'first40_group_state_fingerprints.json',states)
    original=json.loads((base/'predictions.json').read_text());expected=[r for r in original if r['image_id'] in metadata]
    if predictions!=expected:raise ValueError('Actual native output differs from sealed original')
    if not a.prefix40 and sha256(a.out/'predictions.json')!=scene['original_predictions_sha256']:raise ValueError('Full original prediction bytes differ')
    original=iter(read_rows(base/'actual_decisions.jsonl.gz'));actual_calls=0
    for c in read_rows(a.out/'actual_decisions.jsonl.gz'):
        if next(original,None)!=c:raise ValueError('Actual full native solver/threshold/commit differs')
        actual_calls+=1
    if not a.prefix40 and next(original,None) is not None:raise ValueError('Missing complete native call')
    query_keys=[r['key'] for r in read_rows(a.out/'native_queries.jsonl.gz')]
    qsha=hashlib.sha256(json.dumps(query_keys,separators=(',',':')).encode()).hexdigest()
    if not a.prefix40 and (len(query_keys)!=scene['expected_queries'] or qsha!=scene['query_keys_sha256']):raise ValueError('Full native query scope differs')
    if trace.commit_count!=len(trace.observations):raise ValueError('Commit/physical observation coverage differs')
    store=writer.close(dict(scope_sha256=permit['scope_sha256'],capture_sha256=sha256(ROOT/'geo_tools/pirga_full_capture.py'),original_audit_sha256=scene['original_audit_manifest_sha256']),len(query_keys),status='SEALED_INPUTS_PENDING_SUPERVISION')
    initial.update(status='COMPLETE_FULL_NATIVE_INPUTS_PARITY_PASS_PENDING_SUPERVISION' if not a.prefix40 else 'PASS_REAL40_COMPACT_COLLECTOR_PENDING_GATE',
        groups=limit,queries=len(query_keys),calls=trace.next_call,physical_observations=len(trace.observations),actual_descriptor_versions=len(trace.versions),
        actual_commits=trace.commit_count,query_keys_sha256=qsha,predictions_exact_original=True,actual_decisions_exact_original=True,
        full_prediction_bytes_exact=not a.prefix40,indexed_store_manifest_sha256=sha256(a.out/'indexed_inputs/manifest.json'),indexed_store=store,
        body_wall_seconds=time.monotonic()-started,fulltrain_admitted=False,artifacts={str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save_json(a.out/'manifest.json',initial);print(json.dumps({k:initial[k] for k in ['status','scene','groups','queries','calls','physical_observations','actual_descriptor_versions','body_wall_seconds']}),flush=True)

if __name__=='__main__':main()
