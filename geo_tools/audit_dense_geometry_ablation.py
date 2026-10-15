"""Full strict B5/B7 paired attribution; never infer from aggregate equality."""
from common import *
from audit_suite_settings import signature,geo_defaults
from audit_full_prediction_parity import load_prediction


def shared_options(run):
    defaults,digest=geo_defaults(run['commit'])
    if run['source_hashes']['gtr/config.py']!=digest:raise ValueError('unverified config source')
    options={**defaults,**run['geo_options']}
    options.pop('GEOMETRY_WEIGHT')
    if options.get('CALIBRATION_PATH'):options['CALIBRATION_PATH']=dict(sha256=sha256(options['CALIBRATION_PATH']))
    options.setdefault('REUSE_TARGET_FEATURES',False)
    if options['REUSE_TARGET_FEATURES']:raise ValueError('uncached dense ablation required')
    options.pop('TARGET_CACHE_MAX_TARGETS',None) # inactive when reuse is false
    return options


def main():
    sources=[ROOT/'manifests'/name for name in ('causal_B5_dense_seed20261014_strict_suite_v1_sources.json','causal_B7_fixed_k1_strict_suite_v1_sources.json')]
    specs=[json.loads(path.read_text())['video_predictions'] for path in sources]
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    if any(sorted(map(int,spec))!=strict for spec in specs):raise ValueError('complete strict scenes required')
    rows=[]
    for scene in strict:
        paths=[Path(spec[str(scene)]) for spec in specs];loaded=[load_prediction(path) for path in paths]
        predictions=[item[0] for item in loaded];runs=[item[1] for item in loaded];settings=[]
        for index,run in enumerate(runs):
            if not run['causal_online'] or run['video_id']!=scene or run['geo_failures']:raise ValueError('complete successful causal scene required')
            if run['geo_options']['GEOMETRY_WEIGHT']!=[0.,.5][index]:raise ValueError('wrong feature/joint roles')
            setting,verified=signature(run)
            if not verified:raise ValueError('unverified active config defaults')
            # These three fields are the declared geometry-only ablation.
            for key in ('geometry_active','GEOMETRY_WEIGHT','GEOMETRY_METHOD'):setting.pop(key,None)
            settings.append(setting)
        if settings[0]!=settings[1]:raise ValueError('confounded feature/joint configuration')
        shared=[shared_options(run) for run in runs]
        if shared[0]!=shared[1]:raise ValueError('other effective GEO option differs')
        for field in ('images','synchronized_groups','total_vggt_calls'):
            if runs[0][field]!=runs[1][field]:raise ValueError('different input/encoder budget: '+field)
        rows.append(dict(video_id=scene,exact_serialized_predictions=predictions[0]==predictions[1],
            feature_prediction=str(paths[0]),joint_prediction=str(paths[1]),prediction_sha256=[sha256(path) for path in paths],
            manifest_sha256=[sha256(item[2]) for item in loaded],source_commits=[run['commit'] for run in runs],
            shared_effective_options_sha256=hashlib.sha256(json.dumps(shared[0],sort_keys=True).encode()).hexdigest(),
            feature_valid_geometry_calls=runs[0]['geometry_valid_calls'],joint_valid_geometry_calls=runs[1]['geometry_valid_calls'],
            feature_vggt_calls=runs[0]['total_vggt_calls'],joint_vggt_calls=runs[1]['total_vggt_calls']))
    exact=all(row['exact_serialized_predictions'] for row in rows)
    save_json(ROOT/'results/causal_dense_geometry_full_strict_attribution_v1.json',dict(
        status='EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS' if exact else 'GEOMETRY_OUTPUT_DIFFERENCES_OBSERVED',full_strict_suite=True,rows=rows,
        sources_sha256=[sha256(path) for path in sources],
        scope='complete primary-head strict5 causal B5 feature-only versus B7 joint, matched frozen model/data/seed/head/features/scheduler/budgets; exact image/box/score/class/track ID and row order',
        conclusion='No observed independent geometry effect on any serialized output in this complete primary-head suite.' if exact else 'Geometry changes outputs; separate matched-GT correction/harm analysis is needed.',
        limitations='primary trained head only; not three-seed ablation, validated GT pose accuracy, proof all geometry is useless, internal-state equality or speed parity; valid geometry means numerical/background gate only',provenance=provenance()))
    print(json.dumps(dict(full_strict_suite=True,exact_predictions=exact,total_joint_geometry_gate_passes=sum(row['joint_valid_geometry_calls'] for row in rows))))


if __name__=='__main__':main()
