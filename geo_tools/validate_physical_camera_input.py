"""Real-image interface repair check against retained failed-order fixtures."""
import argparse
import collections
import gc
import torch
from common import *
from audit_suite_settings import geo_defaults
from gtr.geometry.physical_camera_input import PhysicalCameraGMT


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=RUNTIME/'camera_order'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    fixture_path=RUNTIME/'camera_order/live_whole_gmt_camera_order_v1/report.json';fixture=json.loads(fixture_path.read_text())
    if fixture['status']!='DIAGNOSTIC_COMPLETED':raise ValueError('complete original negative-order fixture required')
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    result=[]
    for reference in fixture['rows']:
        source=Path(reference['source'])
        if sha256(source)!=reference['source_sha256']:raise ValueError('changed source specification')
        original=reference['variants'][0];canonical=Path(original['path'])
        if sha256(canonical)!=original['sha256']:raise ValueError('changed original canonical fixture')
        prediction=Path(json.loads(source.read_text())['video_predictions']['1']);mp=prediction.parent/'manifest.json';manifest=json.loads(mp.read_text())
        if sha256(mp)!=reference['reference_manifest_sha256'] or sha256(prediction)!=manifest['predictions_sha256']:raise ValueError('changed standalone reference')
        defaults,digest=geo_defaults(manifest['commit'])
        if digest!=manifest['source_hashes']['gtr/config.py']:raise ValueError('unverified reference defaults')
        opts={**defaults,**manifest['geo_options']};opts.setdefault('REUSE_TARGET_FEATURES',False)
        if opts.get('HEAD_PATH') and sha256(opts['HEAD_PATH'])!=manifest['head_checkpoint_sha256']:raise ValueError('changed trained head')
        cfg,model=load_gmt(geo_opts=opts);groups=collections.defaultdict(list)
        for record in video_records(train_data(),1,fixture['max_frames']):groups[record['frame_id']].append(record)
        expected=json.loads(canonical.read_text())
        for variant in ('reverse_every_group','alternate_group_order'):
            seed_all(manifest['seed']);tracker=PhysicalCameraGMT(model);predictions=[]
            try:
                for index,(timestamp,current) in enumerate(sorted(groups.items())):
                    reverse=variant=='reverse_every_group' or index%2==1
                    caller=list(reversed(current)) if reverse else current
                    outputs=tracker.step(map_records(cfg,caller));by_image={r['image_id']:output for r,output in zip(caller,outputs)}
                    # Return positions must correspond to the caller; physical
                    # order below is only for exact reference serialization.
                    for record in current:
                        predictions.extend(custom_instances_to_coco_json(by_image[record['image_id']]['instances'].to('cpu'),record['image_id']))
            finally:model.geo_runtime.close_async()
            path=out/(source.stem+'_'+variant+'.json');save_json(path,predictions)
            row=dict(source=str(source),variant=variant,exact_restored_predictions=predictions==expected,
                actual_vggt_calls=model.geo_runtime.vggt_calls,geometry_valid_calls=model.geo_runtime.geometry_valid_calls,failures=model.geo_runtime.failed,
                predictions=str(path),predictions_sha256=sha256(path),canonical_fixture_sha256=sha256(canonical),standalone_manifest_sha256=sha256(mp))
            result.append(row)
            if not row['exact_restored_predictions'] or row['failures']:raise ValueError('physical-camera input adapter changed canonical reference')
        del tracker,model;gc.collect();torch.cuda.empty_cache()
    save_json(out/'report.json',dict(status='PASS_EXPLICIT_CAMERA_INPUT_CONTRACT',rows=result,full_video=False,full_strict_suite=False,max_frames=fixture['max_frames'],
        negative_order_fixture_sha256=sha256(fixture_path),speed_eligible=False,
        scope='explicit physical-ID input adapter with frozen original tracker; reverse/alternate caller order, restore outputs to caller positions, then exact canonical physical-order serialization against original real scene1 prefix',
        limitations='does not make internal GTA/Hungarian or stochastic query order invariant; original negative-order fixtures retained; no full-video order, valid-pose parity or speed claim; existing production inputs already use canonical physical camera order',provenance=provenance()))


if __name__=='__main__':main()
