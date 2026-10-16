"""Reuse audit of sealed native-only assets. No training or old-score import."""
import argparse
import json
from pathlib import Path
import sqlite3
from common import ROOT,sha256,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    admission_path=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016/fulltrain_admission_v1/ADMISSION.json')
    if sha256(admission_path)!='6815f8d694d7aace041fd5b7ab837af1f50eccae2cf229d191e42148df2e0749':
        raise ValueError('Existing native-only admission changed')
    admission=json.loads(admission_path.read_text());coverage={str(v):dict(video_id=v,native_cache_status='MISSING_REGISTERED_NATIVE_ASSET',training_used_this_round=False) for v in range(1,25)}
    for row in admission['per_scene']:
        scene=row['scene'];spec=row['input_store'];root=Path(spec['root']);mp=root/'manifest.json'
        if sha256(mp)!=spec['manifest_sha256']:raise ValueError('Source manifest drift')
        m=json.loads(mp.read_text())
        if m['native_feature_dim']!=1152 or m['GT_in_input_store'] or m['stored_dtype']!='float16':raise ValueError('Wrong native input semantics')
        for name,digest in m['artifacts'].items():
            if sha256(root/name)!=digest:raise ValueError('Native artifact drift '+str(root/name))
        if (root/'native_vectors.bin').stat().st_size!=m['features']*1152*2:raise ValueError('Truncated physical vectors')
        db=sqlite3.connect('file:'+str(root/'index.sqlite')+'?mode=ro',uri=True)
        times=db.execute('SELECT camera,MIN(frame),MAX(frame),COUNT(*) FROM features GROUP BY camera').fetchall();db.close()
        coverage[str(scene)].update(native_cache_status='SEALED_ORIGINAL_NATIVE_ASSET_SHA_RECHECKED',
            input_root=str(root),manifest_sha256=spec['manifest_sha256'],dimension=1152,stored_dtype=m['stored_dtype'],
            lossy_storage=True,queries=m['queries'],features_including_real_versions=m['features'],source_times=times,
            original_GMT_state_only=True,geometry_or_predictor_outputs=False,artifacts=m['artifacts'])
    root=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1');mp=root/'manifest.json';seal=root/'artifact_sha256.json'
    if sha256(mp)!='a4d2014edcc920f45753e0f76c378bf08f25c22758e3d8770572dc36632c2173' or sha256(seal)!='ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4':
        raise ValueError('Sealed original TRAIN12 native input changed')
    m=json.loads(mp.read_text());pins=json.loads(seal.read_text())['artifacts']
    # Hash input payload only. Existing offline labels/performance files are not
    # adopted as CIR supervision or scientific evidence by this cache inventory.
    verified={}
    for name,digest in pins.items():
        if name.startswith(('native_calls/','native_observations/')) or name=='native_queries.jsonl.gz':
            if sha256(root/name)!=digest:raise ValueError('TRAIN12 native input drift '+name)
            verified[name]=digest
    coverage['12'].update(native_cache_status='SEALED_ORIGINAL_NATIVE_ASSET_SHA_RECHECKED',input_root=str(root),
        manifest_sha256=sha256(mp),input_seal_sha256=sha256(seal),dimension=1152,stored_dtype='float32',lossy_storage=False,
        queries=m['native_queries'],features_including_real_versions=m['native_observations']+m['descriptor_versions'],
        original_GMT_state_only=True,geometry_or_predictor_outputs=False,input_artifacts=verified,
        offline_labels='separate directory, not imported into this CIR inventory')
    result=dict(status='TRAIN24_INVENTORY_NATIVE_CACHE_PARTIAL_NO_CIR_TRAINING',
        native_asset_videos=[int(v) for v,c in coverage.items() if c['native_cache_status'].startswith('SEALED')],
        missing_native_asset_videos=[int(v) for v,c in coverage.items() if c['native_cache_status'].startswith('MISSING')],
        per_video=coverage,training_videos_this_round=0,final24_fit_complete=False,
        source_admission_sha256=sha256(admission_path),
        policy='reuse original native GMT inputs only; do not import past negative results, models, predictions or training logs into CIR scores; missing assets require one continuous causal pass after OFF admission',
        metrics_this_round=None)
    save_json(a.out,result);print(json.dumps({k:v for k,v in result.items() if k!='per_video'}),flush=True)


if __name__=='__main__':main()
