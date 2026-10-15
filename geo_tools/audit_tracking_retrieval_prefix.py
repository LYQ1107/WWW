"""Gate tracking-to-retrieval integration against complete standalone outputs."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--reference-sources',type=Path,nargs='+',required=True)
    parser.add_argument('--report-output',type=Path,required=True)
    parser.add_argument('--full-video',action='store_true',help='verify every output of complete scene streams instead of a short prefix')
    args=parser.parse_args()
    if len(args.runs)!=len(args.reference_sources):raise ValueError('one reference source specification per run')
    rows=[];data=train_data()
    for run,source in zip(args.runs,args.reference_sources):
        manifest=run/'manifest.json';info=json.loads(manifest.read_text())
        spec=json.loads(source.read_text())['video_predictions']
        if not info['causal_online'] or (args.full_video and not info['full_video']) or (not args.full_video and info['max_frames'] is None):
            raise ValueError('requested complete/prefix causal protocol required')
        if info['backbone']=='none' and (info['augmented_queries'] or info['extractors']):
            raise ValueError('tracking metadata leaked into native retrieval features')
        method=info.get('tracking_method','gmt_off');prior=method=='native_fixed_prior'
        scene_rows=[]
        for scene in info['video_ids']:
            prediction=run/f'video_{scene:02d}.json';reference=Path(spec[str(scene)])
            reference_manifest=reference.parent/'manifest.json';ref=json.loads(reference_manifest.read_text())
            if sha256(prediction)!=info['artifacts'][prediction.name] or sha256(reference)!=ref['predictions_sha256']:
                raise ValueError('changed prediction artifact')
            if not ref['full_video'] or not ref['causal_online'] or bool(ref.get('validation_native_output_prior',False))!=prior:
                raise ValueError('reference tracking method/protocol mismatch')
            if method=='vggt_dense':
                if info['tracking_head_checkpoint_sha256']!=ref['head_checkpoint_sha256']:
                    raise ValueError('tracking head mismatch')
                for key,value in info['tracking_options'].items():
                    # Scalar calibration is inactive when a learned head is
                    # loaded; every other explicit tracking option must match.
                    if key!='CALIBRATION_PATH' and ref['geo_options'][key]!=value:
                        raise ValueError('tracking option mismatch: '+key)
            for field in ('seed','checkpoint_sha256','config_sha256','annotation_sha256'):
                if info[field]!=ref[field]:raise ValueError('unpaired tracking inputs: '+field)
            records=video_records(data,scene,info['max_frames']);ids={r['image_id'] for r in records}
            if method=='vggt_dense':
                stats=info['tracking_runtime_stats'][str(scene)]
                if stats['failures'] or stats['vggt_calls']!=len(records):
                    raise ValueError('dense symmetric VGGT prefix was not actually computed for every group')
            actual=json.loads(prediction.read_text())
            expected=[row for row in json.loads(reference.read_text()) if row['image_id'] in ids]
            if prior:
                stats=info['tracking_runtime_stats'][str(scene)]
                if stats['observed_groups']!=len({r['frame_id'] for r in records}) or stats['maximum_extra_vector_columns']!=0:
                    raise ValueError('per-scene tracking runtime counter or zero-column field mismatch')
            scene_rows.append(dict(scene=scene,exact_serialized_standalone_prefix=actual==expected,
                detections=len(actual),prediction_sha256=sha256(prediction),reference_prediction_sha256=sha256(reference),
                reference_manifest_sha256=sha256(reference_manifest)))
        rows.append(dict(run=str(run),manifest_sha256=sha256(manifest),source=str(source),source_sha256=sha256(source),
            tracking_method=method,scenes=scene_rows))
    ok=bool(rows) and all(scene['exact_serialized_standalone_prefix'] for row in rows for scene in row['scenes'])
    strict=set(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    save_json(args.report_output,dict(status='PASS' if ok else 'FAIL',rows=rows,full_video=args.full_video,
        full_strict_suite=args.full_video and all({scene['scene'] for scene in row['scenes']}==strict for row in rows),
        scope=('complete scene outputs' if args.full_video else 'actual multi-scene interleaved image inference prefixes')+' against complete standalone references; exact boxes/scores/classes/image/track IDs and row order; native retrieval receives no added tracking vectors',
        limitations='output equivalence at the stated scope; not retrieval quality, hidden-state equality, runtime repeatability or a substitute for future-prefix causality tests',provenance=provenance()))
    if not ok:raise SystemExit('tracking-to-retrieval outputs differ from standalone tracking')


if __name__=='__main__':main()
