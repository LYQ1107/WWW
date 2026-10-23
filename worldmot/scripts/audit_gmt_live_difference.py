#!/usr/bin/env python3
"""Retain numeric evidence for a real live/cache pre-GTA mismatch."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter,GMT_PYTHON,detection_rows
from worldmot.models.person_detector import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--scene',required=True,type=int)
    p.add_argument('--timestamps',required=True,type=int,nargs='+');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',annotation=a.metadata,split='train',load_annotations=False)
    cache=ImmutableGMTDetectionCache(a.cache);live=GMTDetectorAdapter(python=GMT_PYTHON);rows=[]
    try:
        for pair in ds.synchronized_pairs(a.scene):
            if pair.frame_id not in a.timestamps:continue
            ims={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
            cached=cache.get(a.scene,pair.frame_id,ims,{v:r.image_id for v,r in pair.images.items()})
            actual=live.infer(ims);views=[]
            for v in sorted(cached):
                if len(cached[v])!=len(actual[v]):raise AssertionError('Candidate count differs')
                for c,d in zip(cached[v],actual[v]):
                    cb=np.asarray(c.bbox_xywh,np.float64);db=np.asarray(d.bbox_xywh,np.float64)
                    cr=c.detector_roi_feature;dr=d.detector_roi_feature
                    views.append({'view':v,'detection_id':c.detection_id,'cached_bbox':cb.tolist(),
                        'live_bbox':db.tolist(),'bbox_exact':bool(np.array_equal(cb,db)),
                        'bbox_float32_exact':bool(np.array_equal(cb.astype(np.float32),db.astype(np.float32))),
                        'max_bbox_difference':float(np.abs(cb-db).max()),'score_exact':c.score==d.score,
                        'ROI_exact':bool(np.array_equal(cr,dr)),'max_ROI_difference':float(np.abs(cr-dr).max())})
            rows.append({'timestamp':pair.frame_id,'legacy_all_fields_exact':detection_rows(cached)==detection_rows(actual),'candidates':views})
        if len(rows)!=len(a.timestamps):raise ValueError('Requested timestamp missing')
        all_candidates=[v for r in rows for v in r['candidates']]
        report={'status':'REAL_NUMERIC_DIFFERENCE_AUDIT','scene':a.scene,'frames':rows,
            'all_float32_boxes_scores_ROIs_exact':all(r['bbox_float32_exact'] and r['score_exact'] and r['ROI_exact'] for r in all_candidates),
            'cache_lock_sha256':sha256_file(cache.lock_path),'detector':live.manifest(),
            'GT_in_inference':False,'scope':'Diagnosis only; does not relax candidate or ROI equality'}
        if a.output.exists():raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['frames','detector']}),flush=True)
    finally:live.close()


if __name__=='__main__':main()
