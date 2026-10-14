"""Audit coherent recorded settings across complete native strict5 suites."""
import ast
import functools
import hashlib
from common import *


@functools.lru_cache(None)
def geo_defaults(commit):
    code=subprocess.check_output(['git','show',commit+':gtr/config.py'],cwd=ROOT)
    defaults={}
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        target=ast.unparse(node.targets[0])
        if target.startswith('_C.MODEL.GEO.'):
            defaults[target.rsplit('.',1)[-1]]=ast.literal_eval(node.value)
    return defaults,hashlib.sha256(code).hexdigest()


def signature(run):
    opts=run.get('geo_options') or {}
    defaults,digest=geo_defaults(run['commit'])
    resolved={**defaults,**opts}
    enabled=resolved.get('ENABLED',False) and not run.get('compute_neutral',False)
    feature=enabled and resolved.get('FEATURE_ENABLED',False) and resolved.get('FEATURE_WEIGHT',0)!=0
    geometry=enabled and resolved.get('GEOMETRY_ENABLED',False) and resolved.get('GEOMETRY_WEIGHT',0)!=0
    result={key:run[key] for key in ('checkpoint_sha256','config_sha256','annotation_sha256','seed','causal_online')}
    result.update(feature_active=bool(feature),geometry_active=bool(geometry),
                  corrected_camera=run.get('validation_corrected_gmt_camera_index',False),
                  reversed_encoder=run.get('validation_reversed_vggt_order',False),
                  constant_head_evidence=run.get('validation_constant_head_evidence'),
                  native_output_prior=run.get('validation_native_output_prior',False))
    if feature or geometry:
        result.update({key:resolved.get(key) for key in ('MAX_RESIDUAL','UPDATE_INTERVAL','MAX_GEOMETRY_AGE','FALLBACK_TO_GMT','ASYNC_WORKER')})
        scheduled=resolved.get('SPARSE_UPDATE_ENABLED',False)
        result['SPARSE_UPDATE_ENABLED']=scheduled
        if scheduled:
            result.update({key:resolved.get(key) for key in ('UPDATE_POLICY','MIN_UPDATE_INTERVAL','BUDGET_WINDOW','UPDATE_OFFSET',
                          'SCHEDULER_SEED','COMPUTE_BUDGET','TOTAL_CALL_BUDGET')})
    if feature:
        result.update({key:resolved.get(key) for key in ('FEATURE_WEIGHT','FEATURE_BACKBONE','FEATURE_LAYERS',
                      'POOLING','SYMMETRIZE_VIEW_ORDER','REUSE_TARGET_FEATURES','TARGET_CACHE_MAX_TARGETS')})
        # Before PROJECTION_DIM was introduced, runtime passed FEATURE_DIM
        # directly. Later -1 means exactly that same fallback. Preserve zero
        # as native-dimensional extraction, and compare effective dimensions.
        projection=resolved.get('PROJECTION_DIM',-1)
        result['effective_projection_dim']=resolved['FEATURE_DIM'] if projection<0 else projection
        # Missing later default-off cache fields in older sources are neutral.
        result['REUSE_TARGET_FEATURES']=resolved.get('REUSE_TARGET_FEATURES',False)
        if not result['REUSE_TARGET_FEATURES']:result.pop('TARGET_CACHE_MAX_TARGETS',None)
        if resolved.get('HEAD_PATH'):
            expected=run.get('head_checkpoint_sha256')
            if not expected or sha256(resolved['HEAD_PATH'])!=expected:raise ValueError('head artifact hash mismatch')
            result['head_checkpoint_sha256']=expected
        else:
            result['current_calibration_sha256']=sha256(resolved['CALIBRATION_PATH'])
    if geometry:result.update({key:resolved.get(key) for key in ('GEOMETRY_WEIGHT','GEOMETRY_METHOD')})
    recorded=run.get('source_hashes',{}).get('gtr/config.py')
    verified=not (feature or geometry) or recorded==digest
    return result,verified


def main():
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    baseline=json.loads((ROOT/'manifests/baseline_strict_dev_sources.json').read_text())
    basepath=Path(next(iter(baseline['video_predictions'].values())))
    locked=json.loads(basepath.with_name(basepath.stem+'_manifest.json').read_text())
    rows=[]
    for path in sorted((ROOT/'results').glob('*_metrics.json')):
        report=json.loads(path.read_text())
        if report.get('status')!='PASS' or not report.get('full_video') or report.get('video_ids')!=strict or not report.get('run_manifests'):
            continue
        signatures=[];source_verified=[];runs=[]
        for video,record in sorted(report['run_manifests'].items(),key=lambda row:int(row[0])):
            run_path=Path(record['path'])
            if sha256(run_path)!=record['sha256']:raise ValueError('changed manifest: '+str(run_path))
            run=json.loads(run_path.read_text())
            if not run['full_video'] or run['video_id']!=int(video):raise ValueError('invalid complete scene')
            sig,verified=signature(run);signatures.append(sig);source_verified.append(verified)
            runs.append(dict(video_id=int(video),run_manifest=str(run_path),sha256=sha256(run_path),commit=run['commit'],
                             config_default_snapshot_verified=verified))
        mismatch=[key for key in sorted(set().union(*(s.keys() for s in signatures))) if any(s.get(key)!=signatures[0].get(key) for s in signatures[1:])]
        mismatch += ['locked_'+key for key in ('checkpoint_sha256','config_sha256','annotation_sha256') if signatures[0][key]!=locked[key]]
        rows.append(dict(suite=path.name,metric_sha256=sha256(path),
            status='INCONSISTENT_RECORDED_SETTINGS' if mismatch else 'CONSISTENT_VERIFIED_DEFAULTS' if all(source_verified) else 'CONSISTENT_RECORDED_OPTIONS_DEFAULT_SNAPSHOT_UNVERIFIED',
            mismatch_fields=mismatch,canonical_settings=signatures[0],runs=runs))
    save_json(ROOT/'results/suite_settings_audit_v1.json',dict(status='IN_PROGRESS_SETTINGS_AUDIT',rows=rows,
        inconsistent_suites=sum(bool(row['mismatch_fields']) for row in rows),
        unverified_default_suites=sum('UNVERIFIED' in row['status'] for row in rows),
        scope='complete native strict5 artifacts currently copied into results; exact recorded model/config/head/protocol and active geometry/scheduler settings; inactive geometry options are ignored; historical GEO defaults read from each source commit and checked against recorded source hash when active',
        limitations='Config YAML/default/options consistency is not detection-output equivalence, runtime repeatability or scientific success. Foundation/calibration files without original per-run load hashes can only be checked as current artifacts; no retroactive loaded-weight hash claim. Earlier active runs without a matching recorded config source hash remain explicitly unverified.',provenance=provenance()))
    print(json.dumps(dict(suites=len(rows),inconsistent=[row['suite'] for row in rows if row['mismatch_fields']],
                          unverified=[row['suite'] for row in rows if 'UNVERIFIED' in row['status']])))


if __name__=='__main__':main()
