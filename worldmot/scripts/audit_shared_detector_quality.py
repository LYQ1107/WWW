#!/usr/bin/env python3
"""Offline TRAIN raw-GT detection comparison at fixed native operating points."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.person_detector import sha256_file
from worldmot.training.identity_targets import iou_xywh

VALIDATION=[4,8,12,16,20,24]


def match(boxes,gt,threshold=.5):
    pairs=sorted([(iou_xywh(b,a['bbox']),i,j) for i,b in enumerate(boxes) for j,a in enumerate(gt)],reverse=True)
    used_p=set();used_g=set();matches=[]
    for iou,i,j in pairs:
        if iou<threshold:break
        if i not in used_p and j not in used_g:
            used_p.add(i);used_g.add(j);matches.append((i,j,iou))
    return matches


def rates(counts):
    c=dict(counts)
    c['Precision']=c['TP']/max(1,c['TP']+c['FP'])
    c['Recall']=c['TP']/max(1,c['TP']+c['FN'])
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--gmt-cache',required=True,type=Path)
    p.add_argument('--rtmdet-cache',required=True,type=Path)
    p.add_argument('--annotations',default='/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();data=json.loads(a.annotations.read_text())
    images={r['id']:r for r in data['images'] if r['video_id'] in VALIDATION}
    gt=collections.defaultdict(list)
    for ann in data['annotations']:
        if ann['image_id'] in images:gt[ann['image_id']].append(ann)
    del data
    by_frame=collections.defaultdict(dict)
    for r in images.values():by_frame[r['video_id'],r['frame_id']][r['view_id']]=r['id']
    gmt=ImmutableGMTDetectionCache(a.gmt_cache)
    report={'status':'RUNNING_TRAIN_INTERNAL_DIAGNOSTIC','scope':'six complete held-out TRAIN scenes',
        'IoU':.5,'matching':'greedy descending IoU, one-to-one, identical for both detectors',
        'raw_GT_modified':False,'GT_identity_in_detector_cache':False,'TEST_used':False,
        'annotations_sha256':sha256_file(a.annotations),'GMT_cache_lock_sha256':sha256_file(gmt.lock_path),
        'operating_points':{'GMT':gmt.lock['threshold'],'RTMDet':.25},
        'AP':'NOT_AVAILABLE: fixed operating-point precision/recall diagnostic, not a registered COCO AP evaluation',
        'per_scene':{},'combined':{}}
    totals={m:collections.Counter() for m in ['GMT','RTMDet']}
    for sid in VALIDATION:
        counts={m:collections.Counter() for m in totals}
        rtm={};root=a.rtmdet_cache/f'scene_{sid:02d}'
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='COMPLETE':raise RuntimeError('Original RTMDet object cache incomplete')
        for sh in manifest['shards']:
            path=root/sh['file']
            if sha256_file(path)!=sh['sha256']:raise RuntimeError('Original RTMDet cache checksum mismatch')
            with np.load(path,allow_pickle=False) as d:
                for i,t in enumerate(d['timestamps']):
                    lo,hi=d['frame_offsets'][i:i+2]
                    rtm[int(t)]={v:d['bbox'][int(lo):int(hi)][d['camera_ids'][int(lo):int(hi)]==v].copy() for v in [1,2]}
        camera_counts={m:{v:collections.Counter() for v in [1,2]} for m in totals}
        frame_times=sorted(t for s,t in by_frame if s==sid)
        for t in frame_times:
            candidates=gmt.get(sid,t)
            for view in [1,2]:
                anns=gt[by_frame[sid,t][view]]
                for method,boxes in [('GMT',[d.bbox_xywh for d in candidates[view]]),('RTMDet',rtm[t][view])]:
                    pairs=match(boxes,anns)
                    c={'TP':len(pairs),'FP':len(boxes)-len(pairs),'FN':len(anns)-len(pairs),
                       'candidates':len(boxes),'GT_boxes':len(anns),'images':1}
                    counts[method].update(c);camera_counts[method][view].update(c)
        report['per_scene'][str(sid)]={'timestamps':len(frame_times),
            'detectors':{m:{**rates(c),'per_camera':{str(v):rates(cc) for v,cc in camera_counts[m].items()}} for m,c in counts.items()}}
        for m,c in counts.items():totals[m].update(c)
        print(json.dumps({'scene':sid,**{m:rates(c) for m,c in counts.items()}}),flush=True)
    report['combined']={m:rates(c) for m,c in totals.items()}
    report['status']='COMPLETE_SIX_TRAIN_VALIDATION_DETECTOR_DIAGNOSTIC'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
