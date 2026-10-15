"""Verify old native/Prior references only; never claim new VGGT experiments."""
from common import *


def main():
    rows=[];locked=provenance();strict=[1,10,11,18,19]
    for seed in [20261014,20261015,20261016]:
        for method in ['gmt_off','native_fixed_prior']:
            name=(f'causal_B0_{"" if seed==20261014 else "seed"+str(seed)+"_"}strict_suite_v1_metrics.json' if method=='gmt_off' else f'native_prior_seed{seed}_strict_suite_v1_metrics.json')
            path=ROOT/'results'/name;metrics=json.loads(path.read_text())
            if metrics['status']!='PASS' or not metrics['full_video'] or metrics['video_ids']!=strict:raise ValueError('inherited native full scope differs')
            if bool(metrics.get('validation_native_output_prior'))!=(method=='native_fixed_prior'):raise ValueError('native method reference mislabeled')
            runs=[]
            for scene,item in metrics['run_manifests'].items():
                manifest=Path(item['path'])
                if sha256(manifest)!=item['sha256']:raise ValueError('inherited run changed')
                run=json.loads(manifest.read_text())
                if run['seed']!=seed or not run['causal_online'] or not run['full_video']:raise ValueError('inherited RNG/protocol mismatch')
                for key in ['checkpoint_sha256','config_sha256','annotation_sha256']:
                    if run[key]!=locked[key]:raise ValueError('original native input signature differs')
                if run.get('head_checkpoint_sha256') or run.get('total_vggt_calls',run.get('vggt_calls',0)) or run.get('dinov2_calls',0):raise ValueError('learned/foundation reference excluded')
                runs.append(dict(scene=int(scene),manifest=str(manifest),sha256=sha256(manifest),source_commit=run['commit']))
            for prediction,digest in metrics['prediction_sha256'].items():
                if sha256(prediction)!=digest:raise ValueError('inherited predictions changed')
            for truth,digest in metrics['raw_gt_sha256'].items():
                if sha256(truth)!=digest:raise ValueError('original raw GT changed')
            rows.append(dict(origin='VERIFIED_INHERITED_NATIVE_REFERENCE_NOT_NEW_TRACKVGGT_RESULT',method=method,inference_seed=seed,
                metrics=metrics['metrics'],metric_file=str(path),metric_sha256=sha256(path),runs=runs,
                detector_predictions_SHA_verified=True,raw_GT_SHA_verified=True,images=metrics['images']))
    summary=dict(status='PASS_SIX_INHERITED_NATIVE_REFERENCES',rows=rows,
        scope='Three original native/Prior inference-seed pairs, exact retained weights/config/data signatures and file hashes. These are references, not new teacher/student training or new runtime measurements.',
        current_benchmark_status='Separate fresh native/Prior2/4/10 replay is queued behind named teacher jobs and a real prefix gate; not yet measured.',
        fresh_three_seed_student_comparison='NOT_RUN_NO_VALID_STUDENT',provenance=locked)
    save_json(ROOT/'results/trackvggt_inherited_native_reference_v1.json',summary);print(json.dumps(dict(status=summary['status'],rows=len(rows))))


if __name__=='__main__':main()
