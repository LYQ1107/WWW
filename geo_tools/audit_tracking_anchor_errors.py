"""Past-only GT-anchor analysis of actual identity outputs, after inference."""
import argparse
import collections
import numpy as np
from common import *
from audit_tracking_identity_pairs import predictions,matched_detection_rows


def anchor_states(rows,identity_key,cross_camera=False):
    history=collections.defaultdict(lambda:collections.defaultdict(collections.Counter));seen_gt=set()
    groups=collections.defaultdict(list)
    for index,row in enumerate(rows):groups[row['frame']].append((index,row))
    by_index={}
    for frame,group in sorted(groups.items()):
        for index,row in group:
            identity,truth,camera=row[identity_key],row['gt'],row['camera']
            votes=collections.Counter()
            for view,counts in history.get(identity,{}).items():
                if not cross_camera or view!=camera:votes.update(counts)
            if not votes:
                state='no_past_other_camera_anchor' if cross_camera else 'unanchored_id_gt_previously_matched' if truth in seen_gt else 'unanchored_id_gt_first_matched'
            elif len(votes)>1:state='mixed_past_gt_anchor'
            else:state='consistent_past_gt' if truth in votes else 'conflicting_past_gt'
            by_index[index]=dict(state=state,anchor_gt_counts=dict(votes))
        # Simultaneous camera observations cannot label one another's anchor.
        for _,row in group:
            history[row[identity_key]][row['camera']][row['gt']]+=1;seen_gt.add(row['gt'])
    return [by_index[i] for i in range(len(rows))]


def add_patch_coverage(rows,images):
    import torch
    from gtr.geometry.roi_feature_extractor import PadMapping
    by_image=collections.defaultdict(list)
    for row in rows:by_image[row['image_id']].append(row)
    centers=(np.arange(37,dtype=np.float32)+.5)*14
    for image_id,items in by_image.items():
        image=images[image_id]
        boxes=torch.tensor([row['bbox'] for row in items],dtype=torch.float32)
        boxes[:,2:]+=boxes[:,:2]
        mapped=PadMapping(image['width'],image['height']).boxes(boxes).numpy().clip(0,518)
        counts=(((centers[None,:]>=mapped[:,0,None])&(centers[None,:]<mapped[:,2,None])).sum(1)*
                ((centers[None,:]>=mapped[:,1,None])&(centers[None,:]<mapped[:,3,None])).sum(1))
        for row,count in zip(items,counts):row['patch_center_coverage']=int(count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-sources',type=Path,required=True)
    parser.add_argument('--method-sources',type=Path,required=True)
    parser.add_argument('--video-ids',type=int,nargs='+',required=True)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    required=set(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    if not set(args.video_ids)<=required:raise ValueError('declared clean development only')
    specs=[json.loads(p.read_text())['video_predictions'] for p in (args.baseline_sources,args.method_sources)]
    data=train_data();result=[];raw_hashes={}
    for scene in sorted(args.video_ids):
        paths=[Path(spec[str(scene)]) for spec in specs]
        (base,bm),(method,mm)=[predictions(path) for path in paths]
        for field in ('checkpoint_sha256','config_sha256','annotation_sha256','seed','causal_online'):
            if bm[field]!=mm[field]:raise ValueError('unpaired anchors: '+field)
        if not bm['causal_online']:raise ValueError('causal output linkage analysis only')
        records=video_records(data,scene);images={row['image_id']:row for row in records}
        name=next(video['file_name'] for video in data['videos'] if video['id']==scene)
        truth=collections.defaultdict(list)
        for camera in sorted({row['view_id'] for row in records}):
            path=DATA/'train'/f'{name}_View{camera}'/'gt/gt.txt';raw_hashes[str(path)]=sha256(path);seen=set()
            for line in path.read_text().splitlines():
                values=list(map(float,line.split(',')));frame,identity=map(int,values[:2])
                if (frame,identity) in seen:raise ValueError('duplicate raw GT identity')
                seen.add((frame,identity))
                if values[6]>0 and values[7]==1:truth[(camera,frame)].append((identity,values[2:6]))
        rows=sorted(matched_detection_rows(base,method,images,truth),key=lambda row:(row['frame'],row['camera'],row['image_id'],row['bbox']))
        add_patch_coverage(rows,images);scopes={}
        for cross in (False,True):
            old,new=[anchor_states(rows,key,cross) for key in ('base','new')]
            matrix=collections.Counter();by_coverage=collections.defaultdict(collections.Counter);examples=[];example_counts=collections.Counter()
            for row,previous,current in zip(rows,old,new):
                transition=previous['state']+' -> '+current['state'];matrix[transition]+=1
                coverage=str(row['patch_center_coverage']) if row['patch_center_coverage']<2 else '2+'
                by_coverage[coverage][transition]+=1
                if previous['state']!=current['state'] and example_counts[transition]<2 and len(examples)<50:
                    examples.append(dict(**row,baseline_anchor=previous,method_anchor=current));example_counts[transition]+=1
            scopes['past_other_camera' if cross else 'past_any_camera']=dict(transition_counts=dict(matrix),
                transitions_by_patch_center_coverage={key:dict(value) for key,value in by_coverage.items()},
                conflicting_to_consistent=matrix['conflicting_past_gt -> consistent_past_gt'],
                consistent_to_conflicting=matrix['consistent_past_gt -> conflicting_past_gt'],examples=examples)
        result.append(dict(video_id=scene,common_gt_matched_detections=len(rows),added_detections=len(method.keys()-base.keys()),
                           removed_detections=len(base.keys()-method.keys()),prediction_sha256=list(map(sha256,paths)),
                           patch_center_coverage=dict(collections.Counter(str(row['patch_center_coverage']) if row['patch_center_coverage']<2 else '2+' for row in rows)),scopes=scopes))
    save_json(ROOT/'results'/f'{args.tag}.json',dict(status='PASS_OBSERVABLE_ANCHOR_DIAGNOSTIC',full_strict_suite=set(args.video_ids)==required,
        video_ids=sorted(args.video_ids),rows=result,raw_gt_sha256=raw_hashes,
        scope='actual complete causal output identity links on common one-to-one IoU>=.5 GT-matched detector outputs; all anchors use strictly previous synchronized groups; raw identities scoped per scene; GT only post-inference',
        limitations='Pure/mixed/new anchor categories are observational output relations, not actual Hungarian decisions or proof of appearance/pose/candidate-coverage cause. An ID without a past matched-GT anchor can still have earlier unmatched detections; these categories do not prove a new track was born. Unanchored categories remain separate, not automatically correct or wrong. Patch-center count uses exact official518/14 pad mapping; zero coverage does not imply ROIAlign failure. Past unmatched detections are not labeled. No GT used to choose online pairs, schedules or thresholds.',
        unproven_cause_categories=['appearance similarity','large viewpoint change','geometry failure as cause of identity change','low camera overlap','native candidate coverage insufficiency'],
        source_specs_sha256=[sha256(p) for p in (args.baseline_sources,args.method_sources)],provenance=provenance()))


if __name__=='__main__':main()
