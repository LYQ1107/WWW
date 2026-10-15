"""Bounded real-image diagnostics; no labels, tuning or production changes."""
import argparse
import collections
import gc
import torch
from common import *


def partition(rows):
    names={}
    return [names.setdefault(row['track_id'],len(names)) for row in rows]


def detection_key(row):
    return (row['image_id'],row['category_id'],*row['bbox'],row['score'])


def hook_probe():
    from gtr.geometry.camera_encoding_validation import install_single_image_camera_control
    cfg,model=load_gmt();module=model.roi_heads.s_t_head
    current={};raw=[];observations=[]
    before=module.register_forward_pre_hook(lambda m,inputs:raw.append(inputs[0].detach().clone()))
    correction=install_single_image_camera_control(model)
    def after(m,inputs,output):
        original=raw.pop();actual=inputs[0]
        with torch.inference_mode():unchanged=m.forward(original)
        observations.append(dict(**current,rows=len(actual),original_camera=original[:,-1].unique().tolist(),
            actual_camera=actual[:,-1].unique().tolist(),other_columns_exact=torch.equal(original[:,:-1],actual[:,:-1]),
            output_max_abs_change=float((output-unchanged).abs().max()) if output.numel() else 0.))
    observer=module.register_forward_hook(after)
    rows=video_records(train_data(),1,4);groups=collections.defaultdict(list)
    for row in rows:groups[row['frame_id']].append(row)
    try:
        with torch.inference_mode():
            for timestamp,records in sorted(groups.items()):
                for view,record in enumerate(map_records(cfg,records)):
                    current.update(image_id=record['image_id'],physical_camera=records[view]['view_id'],view=view)
                    model.inference([record],len(records),[1,timestamp,timestamp],view,do_postprocess=False)
        assert len(observations)==len(rows) and {o['view'] for o in observations}=={0,1}
        assert all(o['rows']>0 and o['original_camera']==[0.] and o['actual_camera']==[o['view']/2.] and o['other_columns_exact'] for o in observations)
    finally:before.remove();correction.remove();observer.remove()
    return dict(status='PASS',observations=observations,scope='actual checkpoint and first four synchronized scene1 groups; original ten spatial columns retained; camera column reaches frozen spatial MLP')


def order_probe(source,max_frames,out):
    from audit_suite_settings import geo_defaults
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    reference=Path(json.loads(source.read_text())['video_predictions']['1'])
    manifest=json.loads((reference.parent/'manifest.json').read_text())
    assert sha256(reference)==manifest['predictions_sha256']
    defaults,digest=geo_defaults(manifest['commit'])
    assert manifest['source_hashes']['gtr/config.py']==digest
    opts={**defaults,**manifest['geo_options']};opts.setdefault('REUSE_TARGET_FEATURES',False)
    if opts.get('HEAD_PATH'):assert sha256(opts['HEAD_PATH'])==manifest['head_checkpoint_sha256']
    cfg,model=load_gmt(geo_opts=opts)
    records=video_records(train_data(),1,max_frames);groups=collections.defaultdict(list)
    cameras=sorted({r['view_id'] for r in records});physical={r['image_id']:cameras.index(r['view_id']) for r in records}
    for record in records:groups[record['frame_id']].append(record)
    inference=model.inference
    def preserved_view(batch,view_num,time,view,**kwargs):
        return inference(batch,view_num,time,physical[batch[0]['image_id']],**kwargs)
    model.inference=preserved_view
    variants=[]
    for reverse in (False,True):
        seed_all(manifest['seed']);tracker=CausalGMT(model);predictions=[]
        try:
            for timestamp,current in sorted(groups.items()):
                ordered=list(reversed(current)) if reverse else current
                outputs=tracker.step(map_records(cfg,ordered))
                for record,output in zip(ordered,outputs):
                    predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
        finally:model.geo_runtime.close_async()
        path=out/('reversed.json' if reverse else 'canonical.json');save_json(path,predictions)
        variants.append(dict(path=str(path),sha256=sha256(path),rows=predictions,
            vggt_calls=model.geo_runtime.vggt_calls,geometry_valid_calls=model.geo_runtime.geometry_valid_calls,failures=model.geo_runtime.failed))
    ids=set(physical);expected=[r for r in json.loads(reference.read_text()) if r['image_id'] in ids]
    canonical_matches=variants[0]['rows']==expected
    sorted_rows=[sorted(v['rows'],key=detection_key) for v in variants]
    detector_match=[detection_key(r) for r in sorted_rows[0]]==[detection_key(r) for r in sorted_rows[1]]
    duplicate_keys=any(len({detection_key(r) for r in rows})!=len(rows) for rows in sorted_rows)
    equivalent=detector_match and not duplicate_keys and partition(sorted_rows[0])==partition(sorted_rows[1])
    report=dict(source=str(source),source_sha256=sha256(source),reference_manifest_sha256=sha256(reference.parent/'manifest.json'),
        canonical_matches_standalone_prefix=canonical_matches,identical_detector_membership=detector_match,
        duplicate_detector_keys=duplicate_keys,exact_restored_predictions=sorted_rows[0]==sorted_rows[1],
        same_identity_partition_up_to_global_bijection=equivalent,
        variants=[{k:v for k,v in variant.items() if k!='rows'} for variant in variants])
    if not canonical_matches:raise ValueError('diagnostic canonical path differs from locked standalone prefix')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--max-frames',type=int,default=40);args=parser.parse_args()
    sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    out=RUNTIME/'camera_order'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    hook=hook_probe();gc.collect();torch.cuda.empty_cache()
    save_json(out/'hook.json',dict(**hook,provenance=provenance()))
    rows=[]
    for name in ('causal_B0_strict_suite_v1','causal_B7_fixed_k1_strict_suite_v1'):
        rows.append(order_probe(ROOT/'manifests'/f'{name}_sources.json',args.max_frames,out/name))
        gc.collect();torch.cuda.empty_cache()
    save_json(out/'report.json',dict(status='DIAGNOSTIC_COMPLETED',full_video=False,max_frames=args.max_frames,
        hook=hook,rows=rows,speed_eligible=False,
        scope='reverse whole GMT current camera list, history slot order and sequential association order; preserve image IDs, physical-camera metadata and per-image inference view argument; restore outputs by image ID; compare complete prefix identity partition under one global ID bijection',
        limitations='scene1 bounded prefix only; association ROI jitter consumes RNG in query order, so mismatch includes stochastic order effects; rejected geometry does not establish valid-pose invariance; no GT read or tuning',provenance=provenance()))


if __name__=='__main__':main()
