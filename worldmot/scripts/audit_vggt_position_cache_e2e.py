#!/usr/bin/env python3
"""Verify real delivered identities/point evidence before comparing E2E speed."""
import argparse
import csv
import json
from pathlib import Path
from worldmot.models.person_detector import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();mans=[json.loads((x/'manifest.json').read_text()) for x in (a.before,a.after)]
    for m in mans:
        assert m['status']=='COMPLETE' and m['detector_mode']=='live_verified' and not m['GT_in_inference']
    assert not mans[0]['constant_position_cache'] and mans[1]['constant_position_cache']
    for key in ['source_sha256','settings','scene','frames','metadata_sha256','VGGT_checkpoint_sha256','visible_devices']:
        assert mans[0][key]==mans[1][key],key
    fields=['timestamp','scene_id','tracks_2d_by_camera','point_evidence','point_predictions',
        'identity_merges','identity_aliases','decisions','raw_candidate_count','output_count',
        'lost_ids','deleted_ids','image_ids','detections_by_camera','raw_candidate_shard_sha256']
    streams=[]
    for root,m in zip((a.before,a.after),mans):
        assert sha256_file(root/'tracks.jsonl')==m['prediction_sha256']
        streams.append([json.loads(x) for x in (root/'tracks.jsonl').open()])
    assert len(streams[0])==len(streams[1])==mans[0]['frames']
    for left,right in zip(*streams):
        for field in fields:assert left.get(field)==right.get(field),(left['timestamp'],field)
        for la,ra in zip(left['point_forward_calls'],right['point_forward_calls']):
            assert la['kind']==ra['kind'] and la['image_keys']==ra['image_keys']
        assert len(left['point_forward_calls'])==len(right['point_forward_calls'])
    report={'status':'PASS_REAL_MOT_E2E_SEMANTIC_AND_POINT_EVIDENCE_EXACT_PARITY',
        'frames':mans[0]['frames'],'scene':mans[0]['scene'],'fields_bit_exact':fields,
        'source_sha256':mans[0]['source_sha256'],'GPU_physical_index':mans[0]['visible_devices'],
        'before':{'directory':str(a.before),'manifest_sha256':sha256_file(a.before/'manifest.json'),
            'real_e2e_fps':mans[0]['real_e2e_fps'],'e2e':mans[0]['e2e'],'stages':mans[0]['stages']},
        'after':{'directory':str(a.after),'manifest_sha256':sha256_file(a.after/'manifest.json'),
            'real_e2e_fps':mans[1]['real_e2e_fps'],'e2e':mans[1]['e2e'],'stages':mans[1]['stages'],
            'constant_position_cache':mans[1]['VGGT']['constant_position_cache']},
        'e2e_speedup':mans[1]['real_e2e_fps']/mans[0]['real_e2e_fps'],
        'scope':'Same 32 TRAIN timestamps, exclusive single V100, both live GMT forwards and final ID writes included. Not full-video TEST or an identity improvement.'}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    with a.output.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['mode','stage','count','p50_s','p95_s','total_s','real_e2e_fps'])
        writer.writeheader()
        for mode,m in zip(['original','constant_position_cache'],mans):
            for stage,values in {'rgb_to_final_ID':m['e2e'],**m['stages']}.items():
                writer.writerow({'mode':mode,'stage':stage,**{k:values[k] for k in ['count','p50_s','p95_s','total_s']},
                    'real_e2e_fps':m['real_e2e_fps']})
    print(json.dumps({k:v for k,v in report.items() if k not in ['fields_bit_exact','before','after']}),flush=True)


if __name__=='__main__':main()
