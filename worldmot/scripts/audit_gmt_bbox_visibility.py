#!/usr/bin/env python3
"""Check real bbox-only V1 outputs cannot claim a person segmentation mask."""
import argparse
import json
from pathlib import Path
from worldmot.models.person_detector import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest_path=a.scene_output/'manifest.json';m=json.loads(manifest_path.read_text())
    if m['status']!='COMPLETE' or m['detector_frontend']!='gmt_pre_gta':raise ValueError('Complete GMT V1 output required')
    path=a.scene_output/'tracks.jsonl';rows=[json.loads(x) for x in path.open()]
    eligible=0;all_projection_rows=0
    for row in rows:
        assert row['ablation']=='A_DETECT_FIRST_GMT_PRE_GTA_OMEGA512'
        for dets in row['detections_by_camera'].values():
            assert all(d['mask_area']==0 and d['source']=='gmt_pre_gta_detector' for d in dets)
        for values in row['tracks_2d_projection_by_camera'].values():
            for out in values:
                all_projection_rows+=1
                assert out['native_view_mask'] is False
                if out['evaluation_eligible']:
                    eligible+=1
                    assert out['native_view_detection']
                    assert out['visibility_evidence']=='current_target_detection_bbox'
    if not eligible:raise AssertionError('No current real projection evidence was exercised')
    report={'status':'PASS_REAL_GMT_BBOX_ONLY_VISIBILITY_EVIDENCE','frames':len(rows),
        'eligible_current_detection_projection_rows':eligible,'all_projection_rows':all_projection_rows,
        'false_segmentation_mask_claims':0,'correct_frontend_ablation_label':True,
        'source_sha256':m['source_sha256'],'manifest_sha256':sha256_file(manifest_path),
        'tracks_sha256':sha256_file(path),'scope':'Real inference metadata check; no identity score or FPS claim'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
