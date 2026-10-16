"""Passive missing-TRAIN native inputs, independently sealed before supervision.

Reuses the tested native storage/observer only, not any PIRGA learned method.
Reference predictions are CIR's complete Original GMT actors. No GT is read.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from common import ROOT, CHECKPOINT, load_gmt, train_data, video_records, map_records, seed_all, save_json, sha256
from gtr.geometry.online_runner import CausalGMT
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from pirga_indexed_store import StoreWriter, StoreReader
from pirga_full_capture import FullNativeTrace
from pirga_trace_primitives import CommitDictionary

MISSING=(1,2,3,10,11,14,15,16,18,19,22,23)


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=MISSING,required=True)
    p.add_argument('--reference-runs',type=Path,required=True);p.add_argument('--reference-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage-root',type=Path,required=True);p.add_argument('--prefix40',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.out.parent.resolve()!=a.stage_root.resolve():raise ValueError('Scene output must be inside the bounded stage root')
    a.stage_root.mkdir(parents=True,exist_ok=True)
    if sha256(a.reference_runs)!=a.reference_sha256:raise ValueError('Sealed Original scope changed')
    reference_path=Path(json.loads(a.reference_runs.read_text())['runs'][str(a.video)])
    reference=json.loads(reference_path.read_text())
    if reference['method']!='original' or reference['status']!='COMPLETE' or not reference['full_video'] or reference['video_id']!=a.video:
        raise ValueError('Own complete Original GMT reference required')
    if reference['checkpoint_sha256']!=sha256(CHECKPOINT):raise ValueError('Frozen GMT drift')
    for spec in reference['predictions'].values():
        if sha256(spec['path'])!=spec['sha256']:raise ValueError('Sealed Original predictions changed')
    if shutil.disk_usage(a.out.parent).free<38*(1<<30):raise RuntimeError('Need30GiB reserve plus8GiB bounded stage space')
    all_rows=video_records(train_data(),a.video)
    if sorted(r['image_id'] for r in all_rows)!=sorted(reference['processed_image_ids']):raise ValueError('Complete physical scope changed')
    frames=sorted({r['frame_id'] for r in all_rows});frames=frames[:40] if a.prefix40 else frames
    rows=[r for r in all_rows if r['frame_id'] in set(frames)];metadata={r['image_id']:r for r in rows}
    a.out.mkdir(parents=True);cfg,model=load_gmt();seed_all(reference['seed']);model.min_track_len=0
    if model.cir_controller.enabled or model.geo_runtime.active:raise ValueError('Only pure Original evidence may be collected')
    from cir_configuration import saved_configuration_hashes
    # Compare both after identical serialization/reload. Live CfgNode dumps
    # retain tuple aliases; a loaded saved file does not retain that topology.
    # This changes YAML anchor spelling, not any model setting.
    (a.out/'config.yaml').write_text(cfg.dump())
    if saved_configuration_hashes(a.out/'config.yaml',sha256(CHECKPOINT))['frozen_GMT_config_sha256']!=saved_configuration_hashes(reference_path.parent/'config.yaml',sha256(CHECKPOINT))['frozen_GMT_config_sha256']:
        raise ValueError('Frozen GMT configuration differs from complete Original')
    model.eval().requires_grad_(False);tracker=CausalGMT(model);writer=StoreWriter(a.out/'indexed_inputs',dtype='float32')
    trace=FullNativeTrace(model,a.out,metadata,a.video,writer);trace.tracker=tracker
    tracker.reids=CommitDictionary(trace.commit);trace.install()
    start=time.monotonic();predictions=[];processed=[]
    state=dict(status='RUNNING_NO_SUPERVISION',scene=a.video,GT_in_input_store=False,training=False,full_video=not a.prefix40,
        reference_manifest=dict(path=str(reference_path),sha256=sha256(reference_path)),
        source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in [ROOT/'geo_tools/capture_cir_missing_native.py',ROOT/'geo_tools/pirga_full_capture.py',ROOT/'geo_tools/pirga_indexed_store.py',ROOT/'geo_tools/pirga_trace_primitives.py',ROOT/'gtr/geometry/decision_trace.py',ROOT/'gtr/geometry/online_runner.py',ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py']},
        reference_runs_sha256=a.reference_sha256,checkpoint_sha256=sha256(CHECKPOINT),stored_dtype='float32',
        input_dimension=1152,teacher_calls=0,peer_error_mining_repeated=False)
    save_json(a.out/'manifest.json',state)
    try:
        for i,frame in enumerate(frames):
            current=[r for r in rows if r['frame_id']==frame];trace.completed_images.clear()
            for row in current:
                if sha256(row['file_name'])!=reference['image_sha256'][str(row['image_id'])]:raise ValueError('Actual image bytes changed')
            with torch.inference_mode():output=tracker.step(map_records(cfg,current))
            trace.flush()
            for row,item in zip(current,output):
                predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),row['image_id']));processed.append(row['image_id'])
            if i%20==0:
                used=sum(f.stat().st_size for f in a.stage_root.rglob('*') if f.is_file())
                if used>8*(1<<30) or shutil.disk_usage(a.out).free<30*(1<<30):raise RuntimeError('Bounded storage or disk reserve exceeded')
                state.update(completed_groups=i+1,total_groups=len(frames),queries=trace.query_count,bytes=used)
                save_json(a.out/'manifest.json',state);print(json.dumps(state),flush=True)
        if sorted(processed)!=sorted(metadata):raise ValueError('Physical image coverage differs')
        expected=[r for r in json.loads(Path(reference['predictions']['raw']['path']).read_text()) if r['image_id'] in metadata]
        if predictions!=expected:raise ValueError('Observer changed actual Original GMT predictions')
        if trace.commit_count!=len(trace.observations):raise ValueError('Native commit/physical observation coverage mismatch')
    except BaseException as error:
        state.update(status='FAILED_NO_ADMISSION',reason=repr(error));save_json(a.out/'manifest.json',state);raise
    finally:trace.close()
    torch.cuda.synchronize();save_json(a.out/'predictions.json',predictions)
    manifest=writer.close(dict(reference_manifest_sha256=sha256(reference_path),capture_sha256=sha256(ROOT/'geo_tools/pirga_full_capture.py')),
        trace.query_count,status='SEALED_INPUTS_PENDING_SUPERVISION')
    with_reader=StoreReader(a.out/'indexed_inputs',sha256(a.out/'indexed_inputs/manifest.json'));with_reader.db.close()
    state.update(status='PASS_REAL40_MISSING_INPUT_CAPTURE' if a.prefix40 else 'COMPLETE_MISSING_NATIVE_INPUTS_PARITY_PASS_PENDING_SUPERVISION',
        groups=len(frames),images=len(processed),queries=trace.query_count,calls=trace.next_call,
        physical_observations=len(trace.observations),actual_commits=trace.commit_count,
        indexed_store=manifest,indexed_store_manifest_sha256=sha256(a.out/'indexed_inputs/manifest.json'),
        predictions_exact_original=True,actual_decisions_exact_original=None,
        decision_reference_note='Own formal Original runs retain predictions but no passive native decision trace; output equality does not assert unavailable call-by-call reference equality.',
        wall_seconds=time.monotonic()-start,full24_training_admitted=False,
        artifacts={str(f.relative_to(a.out)):sha256(f) for f in a.out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    save_json(a.out/'manifest.json',state)


if __name__=='__main__':main()
