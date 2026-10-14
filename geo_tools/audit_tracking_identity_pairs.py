"""All-time identity-pair corrections on common, GT-matched detections."""
import argparse
import collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from evaluate_live_retrieval import iou_matrix


def count_equal_pairs(rows,fields,cross_camera):
    counts=collections.Counter(tuple(row[field] for field in fields) for row in rows)
    choose=lambda n:n*(n-1)//2
    total=sum(choose(n) for n in counts.values())
    if cross_camera:
        within=collections.Counter(tuple(row[field] for field in [*fields,'camera']) for row in rows)
        total-=sum(choose(n) for n in within.values())
    return total


def pair_outcomes(rows,cross_camera=False):
    total=count_equal_pairs(rows,[],cross_camera)
    positive=count_equal_pairs(rows,['gt'],cross_camera)
    base_tp=count_equal_pairs(rows,['gt','base'],cross_camera)
    new_tp=count_equal_pairs(rows,['gt','new'],cross_camera)
    base_fp=count_equal_pairs(rows,['base'],cross_camera)-base_tp
    new_fp=count_equal_pairs(rows,['new'],cross_camera)-new_tp
    both_tp=count_equal_pairs(rows,['gt','base','new'],cross_camera)
    both_fp=count_equal_pairs(rows,['base','new'],cross_camera)-both_tp
    return dict(detections=len(rows),unordered_pairs=total,gt_positive_pairs=positive,
        baseline=dict(TP=base_tp,FP=base_fp,FN=positive-base_tp,TN=total-positive-base_fp),
        method=dict(TP=new_tp,FP=new_fp,FN=positive-new_tp,TN=total-positive-new_fp),
        fixes_false_negative_pairs=new_tp-both_tp,fixes_false_positive_pairs=base_fp-both_fp,
        harms_true_positive_pairs=base_tp-both_tp,harms_true_negative_pairs=new_fp-both_fp)


def detection_key(row):
    return (int(row['image_id']),int(row['category_id']),float(row['score']),*map(float,row['bbox']))


def predictions(path):
    data=json.loads(path.read_text());result={}
    for row in data:
        key=detection_key(row)
        if key in result:raise ValueError('duplicate exact detector key; common detection identity is ambiguous')
        result[key]=row
    manifest=path.with_name(path.stem+'_manifest.json')
    if not manifest.exists():manifest=path.parent/'manifest.json'
    info=json.loads(manifest.read_text())
    if not info.get('full_video'):raise ValueError('complete videos required')
    if info.get('prediction_sha256',info.get('predictions_sha256'))!=sha256(path):raise ValueError('changed predictions')
    return result,info


def matched_detection_rows(base,method,images,ground_truth):
    """GT-match common detector outputs once; labels never enter tracking."""
    by_image=collections.defaultdict(list)
    for key in sorted(base.keys()&method.keys()):by_image[key[0]].append(key)
    matched=[]
    for image_id,keys in by_image.items():
        image=images[image_id];truth=ground_truth[(image['view_id'],image['frame_id'])]
        if not truth:continue
        overlaps=iou_matrix([base[key]['bbox'] for key in keys],[row[1] for row in truth])
        ri,ci=linear_sum_assignment(np.where(overlaps>=.5,1-overlaps,1e6))
        for i,j in zip(ri,ci):
            if overlaps[i,j]>=.5:
                key=keys[i]
                matched.append(dict(gt=truth[j][0],camera=image['view_id'],frame=image['frame_id'],image_id=image_id,
                                    bbox=base[key]['bbox'],base=base[key]['track_id'],new=method[key]['track_id']))
    return matched


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-sources',type=Path,required=True);parser.add_argument('--method-sources',type=Path,required=True);parser.add_argument('--tag',required=True);parser.add_argument('--video-ids',type=int,nargs='+');parser.add_argument('--report-output',type=Path);args=parser.parse_args()
    sources=[json.loads(path.read_text())['video_predictions'] for path in [args.baseline_sources,args.method_sources]]
    full_required=set(map(str,json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids']))
    required=set(map(str,args.video_ids)) if args.video_ids else full_required
    if not required<=full_required or any(not required<=set(source) for source in sources):raise ValueError('requested declared strict scenes missing')
    if not args.video_ids and any(set(source)!=required for source in sources):raise ValueError('all strict development scenes required by default')
    data=train_data();rows=[];gt_hashes={}
    for scene in sorted(map(int,required)):
        paths=[Path(source[str(scene)]) for source in sources]
        (base,base_info),(method,method_info)=[predictions(path) for path in paths]
        for field in ['causal_online','checkpoint_sha256','config_sha256','annotation_sha256','seed']:
            if base_info[field]!=method_info[field]:raise ValueError('mixed tracking protocols/checkpoints/configs')
        records=video_records(data,scene);images={record['image_id']:record for record in records}
        scene_name=next(video['file_name'] for video in data['videos'] if video['id']==scene)
        ground_truth=collections.defaultdict(list)
        for camera in sorted({record['view_id'] for record in records}):
            path=DATA/'train'/f'{scene_name}_View{camera}'/'gt/gt.txt';gt_hashes[str(path)]=sha256(path);seen=set()
            for line in path.read_text().splitlines():
                fields=list(map(float,line.split(',')));frame,identity=map(int,fields[:2])
                if (frame,identity) in seen:raise ValueError('duplicate raw GT identity')
                seen.add((frame,identity))
                if fields[6]>0 and fields[7]==1:ground_truth[(camera,frame)].append((identity,fields[2:6]))
        common=base.keys()&method.keys()
        matched=matched_detection_rows(base,method,images,ground_truth)
        rows.append(dict(video_id=scene,baseline_prediction=str(paths[0]),method_prediction=str(paths[1]),prediction_sha256=[sha256(path) for path in paths],
            baseline_detections=len(base),method_detections=len(method),common_detections=len(common),added_detections=len(method.keys()-base.keys()),removed_detections=len(base.keys()-method.keys()),
            all_time_identity_pairs=pair_outcomes(matched),cross_camera_all_time_identity_pairs=pair_outcomes(matched,True)))
    save_json(args.report_output or ROOT/'results'/f'{args.tag}.json',dict(status='PASS',video_ids=sorted(map(int,required)),full_strict_suite=required==full_required,
        scope='explicit complete video scenes; exact detector-key intersection; independent raw-GT one-to-one IoU>=0.5 matching of common detections; partial scene selection is not a full-suite claim',
        semantics='unordered detection identity-pair classification across all timestamps, with separate cross-camera-only counts; corrections/harms are pair relations, not counts of Hungarian decisions or a replacement for HOTA/CVIDF1',
        limitations='added/removed detections excluded from identity-pair counts and reported separately; long tracks contribute quadratically many pairs; raw GT uncertainty is unchanged',
        sources_sha256=[sha256(path) for path in [args.baseline_sources,args.method_sources]],raw_gt_sha256=gt_hashes,rows=rows,provenance=provenance()))


if __name__=='__main__':main()
