#!/usr/bin/env python3
"""Posthoc stage evidence on full valid TEST videos; not a metric substitute."""
from __future__ import annotations
import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from worldmot.evaluation.official_fullvideo import read_mot, sha
from worldmot.training.identity_targets import iou_xywh
from cache_fulltrain import atomic_json


def matches(boxes,targets,threshold=.5):
    if not boxes or not targets:return {}
    overlap=np.asarray([[iou_xywh(b['bbox_xywh'],g[2:6]) for g in targets] for b in boxes])
    rr,cc=linear_sum_assignment((overlap<threshold)*1e6+1-overlap)
    return {int(targets[j][1]):boxes[i] for i,j in zip(rr,cc) if overlap[i,j]>=threshold}


def eligible(rows):
    return [r for r in rows if r.get('evaluation_eligible',True) and r.get('visible',True) and r.get('bbox_xywh') is not None]


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def render_case(case,metadata,root,output):
    from PIL import Image,ImageDraw,ImageFont
    lookup={(r['video_id'],r['frame_id'],r['view_id']):r for r in metadata['images']}
    palette=['#ff6262','#57d9ff','#f8d45c','#b987ff','#63f2b2','#ff8bd4']
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    images=[]
    for view in [1,2]:
        rec=lookup[(case['scene'],case['timestamp'],view)]
        image=Image.open(Path(root)/'test'/rec['file_name']).convert('RGB');draw=ImageDraw.Draw(image)
        for gt in case['GT_by_camera'].get(str(view),[]):
            x,y,w,h=gt[2:6];draw.rectangle((x,y,x+w,y+h),outline='#ffffff',width=2)
            draw.text((x,y+h),f'GT {int(gt[1])}',fill='#ffffff',font=font,stroke_width=2,stroke_fill='black')
        for obj in case['predictions_by_camera'].get(str(view),[]):
            x,y,w,h=obj['bbox_xywh'];gid=int(obj['global_id']);color=palette[gid%len(palette)]
            draw.rectangle((x,y,x+w,y+h),outline=color,width=5)
            draw.text((x,max(0,y-24)),f'ID {gid}',fill=color,font=font,stroke_width=2,stroke_fill='black')
        draw.rectangle((0,0,image.width,34),fill='black')
        draw.text((8,4),f'{case["method"]} | Scene {case["scene"]} | t={case["timestamp"]} | Camera {view}',fill='white',font=font)
        image.thumbnail((960,540));images.append(image)
    mosaic=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)))
    offset=0
    for image in images:mosaic.paste(image,(offset,0));offset+=image.width
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True);mosaic.save(target,quality=88)
    return str(target)


def analyze(root,output,render=False):
    root=Path(root);out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);audit=json.loads((root/'audit/data_audit.json').read_text())['test']
    metadata=json.loads((root/'audit/test_metadata.json').read_text());methods={};samples=[]
    logs=root/'test_failure_events';logs.mkdir(exist_ok=True)
    for evaluation_path in sorted((root/'test_evaluation').glob('*/metrics.json')):
        evaluation=json.loads(evaluation_path.read_text())
        if evaluation.get('status')!='COMPLETE_VALID_SCOPE':continue
        label=evaluation_path.parent.name;actual='v1' if label=='v1_observed_bbox' else label
        box_source=evaluation.get('box_source','tracks_2d_by_camera');counts=collections.Counter();rejections=collections.Counter()
        chosen=collections.Counter();scene_rows=[];event_path=logs/(label+'.jsonl.gz')
        with gzip.open(event_path,'wt') as event_log:
            for scene in audit['scenes']:
                sid=scene['scene'];directory=root/'test'/actual/f'scene_{sid:02d}'
                manifest=json.loads((directory/'manifest.json').read_text())
                if manifest.get('status')!='COMPLETE' or not manifest.get('full_video'):raise RuntimeError('partial failure-analysis input')
                groundtruth={};previous={};local=collections.Counter()
                if scene['official_gt_legal']:
                    for raw in scene['raw_gt']:
                        if sha(raw['path'])!=raw['sha256']:raise RuntimeError('raw GT changed after audit')
                        by_time=collections.defaultdict(list)
                        for row in read_mot(raw['path']):
                            if row[6]==0 or (len(row)>7 and int(row[7])!=1):continue
                            by_time[int(row[0])].append(row)
                        groundtruth[str(raw['view'])]=by_time
                for line in (directory/'tracks.jsonl').open():
                    row=json.loads(line);timestamp=int(row['timestamp']);local['timestamps']+=1
                    alignment=row.get('alignment',{});aligned=alignment.get('success')
                    if aligned is False:local['world_alignment_failed_timestamps']+=1
                    qr=row.get('query_runtime',{})
                    local['query_births']+=qr.get('created_queries',0);local['query_capacity_overflow']+=qr.get('query_capacity_overflow',0)
                    for rejected in qr.get('rejections',[]):rejections[rejected['reason']]+=1
                    for observation in row.get('world_observations',[]):
                        if 'diagnostics' in observation:
                            local['world_observations']+=1
                            if not observation['diagnostics'].get('geometry_coordinate_valid',False):local['geometry_invalid_observations']+=1
                    if not groundtruth:continue
                    events=[];mapping={};gt_by_camera={};pred_by_camera={}
                    for view,gt_by_time in groundtruth.items():
                        targets=gt_by_time.get(timestamp,[]);preds=eligible(row[box_source].get(view,[]));gt_by_camera[view]=targets;pred_by_camera[view]=preds
                        matched=matches(preds,targets);mapping[view]=matched;local['matched_GT_observations']+=len(matched)
                        local['unmatched_prediction_observations_diagnostic']+=len(preds)-len(matched)
                        dets=row.get('detections_by_camera',{}).get(view)
                        detected=matches(dets,targets) if dets is not None else None
                        projection_control=matches(eligible(row.get('tracks_2d_observation_boxes_by_camera',{}).get(view,[])),targets)
                        for gt in targets:
                            identity=int(gt[1]);obj=matched.get(identity)
                            if obj is None:
                                local['unmatched_GT_observations_diagnostic']+=1
                                if detected is None:reason='UPSTREAM_DETECTION_UNAVAILABLE_NATIVE_GMT'
                                elif identity not in detected:reason='DETECTOR_MISS_AT_IOU_0_5'
                                elif identity in projection_control and label=='v1':reason='V1_PROJECTION_BOX_FAILURE'
                                elif actual=='v1' and aligned is False:reason='WORLD_ALIGNMENT_ABSTENTION'
                                else:reason='DETECTION_EXISTS_BUT_NO_MATCHING_OUTPUT'
                                events.append({'type':reason,'view':view,'GT_identity':identity,'GT_bbox':gt[2:6]})
                            else:
                                key=(view,identity);gid=int(obj['global_id']);old=previous.get(key)
                                if old and old[0]!=gid:
                                    events.append({'type':'PERSISTENT_ID_CHANGE_DIAGNOSTIC','view':view,'GT_identity':identity,
                                        'previous_ID':old[0],'current_ID':gid,'gap_timestamps':timestamp-old[1]})
                                previous[key]=(gid,timestamp)
                    if '1' in mapping and '2' in mapping:
                        for identity in mapping['1'].keys() & mapping['2'].keys():
                            first=mapping['1'][identity];second=mapping['2'][identity]
                            local['same_GT_cross_view_opportunities']+=1
                            if first['global_id']!=second['global_id']:
                                events.append({'type':'SAME_PERSON_SPLIT_ACROSS_CAMERAS','GT_identity':identity,
                                    'IDs':[first['global_id'],second['global_id']]})
                            else:local['same_GT_cross_view_shared_ID']+=1
                        first={int(p['global_id']):g for g,p in mapping['1'].items()};second={int(p['global_id']):g for g,p in mapping['2'].items()}
                        for gid in first.keys() & second.keys():
                            if first[gid]!=second[gid]:events.append({'type':'DIFFERENT_PEOPLE_SHARE_ID_ACROSS_CAMERAS','global_id':gid,'GT_identities':[first[gid],second[gid]]})
                    for event in events:
                        local[event['type']]+=1
                        if chosen[event['type']]<2:
                            chosen[event['type']]+=1
                            samples.append(finite_json({'method':label,'scene':sid,'timestamp':timestamp,'event':event,
                                'GT_by_camera':gt_by_camera,'predictions_by_camera':pred_by_camera,'alignment':alignment,
                                'query_rejections':qr.get('rejections',[]),'image_ids':row.get('image_ids',{})}))
                    if events:event_log.write(json.dumps(finite_json({'scene':sid,'timestamp':timestamp,'events':events}),allow_nan=False)+'\n')
                counts.update(local);scene_rows.append({'scene':sid,'GT_diagnostics_legal':scene['official_gt_legal'],'counts':dict(local)})
        methods[label]={'counts':dict(counts),'query_rejection_reasons':dict(rejections),'per_scene':scene_rows,
            'all_failure_events':str(event_path),'all_failure_events_sha256':sha(event_path),
            'official_IDSW':evaluation['TrackEval']['combined']['IDSW']}
    if render:
        rendered=collections.Counter()
        for index,case in enumerate(samples):
            if rendered[case['method']]>=3:continue
            case['visualization']=render_case(case,metadata,'/data/DATASETS/TRACKING/JDE/VisionTrack',
                out.parent/'worldmot_v2_test_failure_visuals'/f'{case["method"]}_{index:03d}.jpg')
            rendered[case['method']]+=1
    report={'status':'POSTHOC_FULL_VALID_VIDEO_DIAGNOSTICS','diagnostic_IoU':.5,
        'matching_is_not_TrackEval_or_official_IDSW':True,'official_IDSW_source':'unchanged TrackEval CLEAR',
        'attribution_limit':'Stage evidence identifies missing input/output and ID inconsistencies; a unique causal origin for an ID change is not inferred from correlation.',
        'invalid_raw_GT_scenes_excluded_from_identity_diagnostics':True,'methods':methods,'selected_failure_cases':samples,
        'case_selection':'first two events of each category per method; all event counts and full gzip logs retained',
        'raw_GT_modified':False}
    atomic_json(out,report);return {'methods':len(methods),'selected_cases':len(samples),'output':str(out)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
    p.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'results/worldmot_v2_failure_analysis.json'));p.add_argument('--render',action='store_true');a=p.parse_args()
    print(json.dumps(analyze(a.root,a.output,a.render),indent=2))
