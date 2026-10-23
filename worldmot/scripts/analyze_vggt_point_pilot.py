#!/usr/bin/env python3
"""Post-hoc TRAIN identity diagnostics; never presented as official MOT scores."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.person_detector import PersonDetection,sha256_file
from worldmot.training.identity_targets import match_detections_to_gt


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--render',action='store_true')
    a=p.parse_args();ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train')
    counts=collections.Counter();history={};last_gid={};snapshots=[];per_scene={}
    for manifest_path in sorted(a.predictions.glob('scene_*/manifest.json')):
        manifest=json.loads(manifest_path.read_text())
        if manifest['split']!='train' or manifest['status']!='COMPLETE':raise ValueError('Completed TRAIN pilots only')
        sid=manifest['scene'];scene_counts=collections.Counter()
        track_path=manifest_path.with_name('tracks.jsonl')
        if sha256_file(track_path)!=manifest['prediction_sha256']:raise RuntimeError('Frozen prediction changed')
        for line in track_path.open():
            row=json.loads(line);t=row['timestamp']
            dets={int(v):[PersonDetection(int(v),d['detection_id'],tuple(d['bbox_xywh']),d['score']) for d in values]
                  for v,values in row['detections_by_camera'].items()}
            gt={int(v):ds.ground_truth(int(image_id)) for v,image_id in row['image_ids'].items()}
            targets=match_detections_to_gt(dets,gt,sid)
            by_det={};by_camera=collections.defaultdict(dict)
            raw_gt_by_camera={v:{targets[v,i].identity_key for i in range(len(values))
                                if targets[v,i].identity_key is not None} for v,values in dets.items()}
            scene_counts['raw_candidates']+=sum(map(len,dets.values()))
            for v,values in row['tracks_2d_by_camera'].items():
                view=int(v)
                for out in values:
                    detection_id=out.get('detection_id')
                    if detection_id is None:
                        index=min(range(len(dets[view])),key=lambda i:np.abs(np.asarray(dets[view][i].bbox_xywh)-out['bbox_xywh']).max())
                        if not np.allclose(dets[view][index].bbox_xywh,out['bbox_xywh'],atol=1e-4):
                            raise ValueError('Primary output does not use a shared current detection box')
                        detection_id=dets[view][index].detection_id
                    else:index=next(i for i,d in enumerate(dets[view]) if d.detection_id==detection_id)
                    target=targets[view,index].identity_key;gid=out['global_id']
                    by_det[view,detection_id]=(target,gid)
                    history[sid,t,view,gid]=target
                    scene_counts['output_boxes']+=1
                    if target is not None:
                        scene_counts['GT_matched_output_boxes']+=1;by_camera[view][target]=gid
                        key=(sid,view,target)
                        if key in last_gid and last_gid[key]!=gid:scene_counts['diagnostic_temporal_ID_changes']+=1
                        last_gid[key]=gid
                    else:scene_counts['unmatched_or_ambiguous_output_boxes']+=1
            if 1 in raw_gt_by_camera and 2 in raw_gt_by_camera:
                for target in raw_gt_by_camera[1]&raw_gt_by_camera[2]:
                    scene_counts['same_GT_both_views_detected']+=1
                    if target in by_camera[1] and target in by_camera[2] and by_camera[1][target]==by_camera[2][target]:
                        scene_counts['correct_cross_view_shared_ID']+=1
                for ta,ga in by_camera[1].items():
                    for tb,gb in by_camera[2].items():
                        if ga==gb and ta!=tb:scene_counts['different_GT_cross_view_wrong_merge']+=1
            for ev in row.get('point_evidence',[]):
                if 'valid_point_count' not in ev:continue
                scene_counts['candidate_point_evidence_pairs']+=1
                scene_counts['valid_point_votes']+=ev['valid_point_count']
                scene_counts['inside_box_point_votes']+=ev['inside_box_count']
                if ev['association_score']<=0:continue
                target=by_det.get((ev['target_camera'],ev['detection_id']),(None,None))[0]
                if ev['kind']=='TEMPORAL':
                    source=history.get((sid,ev['reference_timestamp'],ev['source_camera'],ev['global_id']))
                else:source=by_det.get((ev['source_camera'],ev['source_detection_id']),(None,None))[0]
                if source is not None and target is not None:
                    scene_counts['accepted_point_pairs_with_GT_identity']+=1
                    scene_counts['GT_consistent_accepted_point_pairs' if source==target else 'GT_inconsistent_accepted_point_pairs']+=1
            if row.get('point_predictions') and len(snapshots)<4:snapshots.append((sid,row))
        counts.update(scene_counts);per_scene[str(sid)]={'counts':dict(scene_counts),
            'real_e2e_fps':manifest['real_e2e_fps'],'stages':manifest['stages'],
            'manifest_sha256':sha256_file(manifest_path),'prediction_sha256':manifest['prediction_sha256']}
    report={'status':'COMPLETE_TRAIN_PILOT_DIAGNOSTIC','scope':'post-hoc TRAIN at fixed IoU .5',
        'official_TEST_metrics':False,'raw_GT_modified':False,'counts':dict(counts),'per_scene':per_scene,
        'cross_view_same_ID_rate':counts['correct_cross_view_shared_ID']/counts['same_GT_both_views_detected'] if counts['same_GT_both_views_detected'] else None,
        'point_pair_identity_consistency':counts['GT_consistent_accepted_point_pairs']/counts['accepted_point_pairs_with_GT_identity'] if counts['accepted_point_pairs_with_GT_identity'] else None,
        'point_accuracy_definition':'GT identity consistency of accepted box-level support, not dense point correspondence ground truth'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if a.render:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        for sid,row in snapshots:
            fig,axes=plt.subplots(1,2,figsize=(16,6))
            for ax,view in zip(axes,[1,2]):
                image=ds.images[int(row['image_ids'][str(view)])]
                ax.imshow(Image.open(ds.image_path(image)).convert('RGB'))
                for o in row['tracks_2d_by_camera'][str(view)]:
                    x,y,w,h=o['bbox_xywh'];color=plt.get_cmap('tab20')(o['global_id']%20)
                    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=2))
                    ax.text(x,y,f"ID {o['global_id']}",color='white',bbox={'facecolor':color,'alpha':.8})
                for prediction in row['point_predictions']:
                    keys=prediction['image_keys']
                    for k,key in enumerate(keys):
                        if key==[view,row['timestamp']]:
                            pts=np.asarray(prediction['predicted_xy'][k]);valid=np.asarray(prediction['valid'][k],bool)
                            vis=np.asarray(prediction['visibility'][k]);conf=np.asarray(prediction['confidence'][k])
                            good=valid&(vis>=.5)&(conf>=.5)
                            if good.any():ax.scatter(pts[good,0],pts[good,1],s=8,c='yellow',alpha=.7)
                ax.set_title(f"TRAIN scene {sid} frame {row['timestamp']} camera {view}: predicted IDs / confident points")
                ax.axis('off')
            fig.tight_layout();fig.savefig(a.output.with_name(a.output.stem+f"_scene{sid}_frame{row['timestamp']}.png"),dpi=140)
            plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='per_scene'}),flush=True)


if __name__=='__main__':main()
