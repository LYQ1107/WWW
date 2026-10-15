"""Freeze every legal fit scene/image and existing native trace before export."""
import argparse
import hashlib
import json
from pathlib import Path
from common import ROOT,train_data,video_records,sha256,save_json
from audit_pirga_fulltrain import FIT,checked,read_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--prefix-gate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    gate=json.loads(a.prefix_gate.read_text())
    if gate['status']!='PASS_REAL_FIT40_OFF_ON_NATIVE_EXPORT':raise ValueError('Actual prefix gate required')
    pins=json.loads(a.source_manifest.read_text())['artifacts'];data=train_data();scenes=[]
    for scene in FIT:
        base=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')/f'scene{scene}_full_v1';mp=base/'manifest.json';checked(mp,pins[str(mp)])
        m=json.loads(mp.read_text());images=video_records(data,scene)
        if m['status']!='COMPLETE_FULL_CAUSAL_AUDIT' or m['groups']!=len({r['frame_id'] for r in images}) or m['images']!=len(images):raise ValueError('Incomplete authorized scene')
        keys=[]
        checked(base/'actual_decisions.jsonl.gz',pins[str(base/'actual_decisions.jsonl.gz')])
        checked(base/'predictions.json',pins[str(base/'predictions.json')])
        for c in read_rows(base/'actual_decisions.jsonl.gz'):
            if c['stage']!='run_memory_tracker':keys.extend([[scene,*q] for q in c['queries']])
        hashes={str(r['image_id']):sha256(r['file_name']) for r in images}
        scenes.append(dict(scene=scene,groups=m['groups'],images=images,image_sha256=hashes,expected_queries=len(keys),
            query_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),
            original_audit_manifest=str(mp),original_audit_manifest_sha256=pins[str(mp)],
            original_predictions_sha256=pins[str(base/'predictions.json')],original_decisions_sha256=pins[str(base/'actual_decisions.jsonl.gz')],
            historical_native_seconds=m['inference_seconds']))
        print(json.dumps(dict(frozen_scene=scene,groups=m['groups'],queries=len(keys))),flush=True)
    files=['geo_tools/pirga_full_capture.py','geo_tools/export_pirga_full_scene.py','geo_tools/run_pirga_full_export.py','geo_tools/pirga_indexed_store.py','geo_tools/pirga_trace_primitives.py',
           'geo_tools/common.py','gtr/geometry/decision_trace.py','gtr/geometry/online_runner.py','gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/roi_heads/gtr_roi_heads.py','configs/VISION_test.yaml']
    save_json(a.out,dict(status='FROZEN_CONTINUOUS_ALL11_INPUT_SCOPE',schema='pirga-full-native-export/v1',fit_scenes=list(FIT),development_excluded=[12],
        groups_start=1,selection='all actual primary queries in every complete fit scene, no GT/frame/error filtering',
        prefix_gate_path=str(a.prefix_gate),prefix_gate_sha256=sha256(a.prefix_gate),source_manifest_sha256=sha256(a.source_manifest),
        bound_capture_and_native_source_sha256={f:sha256(ROOT/f) for f in files},scenes=scenes,expected_total_queries=sum(x['expected_queries'] for x in scenes),
        original_checkpoint_sha256='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8',
        inference_uses_GT_labels=False,source_GT_recomputed=False,full_image_hash_freeze_CPU_only=True))

if __name__=='__main__':main()
