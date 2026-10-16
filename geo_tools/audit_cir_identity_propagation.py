"""Offline causal-owner error episodes on complete mutated-state predictions.

Never imported by the controller. UNKNOWN/unanchored is not a repaired identity.
Camera-frame observation durations are not invented physical seconds.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import train_data,video_records,save_json,sha256
from evaluate_cir_fulltrain import collect


def iou_matrix(boxes,gt):
    a=np.asarray(boxes,dtype=float).reshape(-1,4);b=np.asarray(gt,dtype=float).reshape(-1,4)
    upper=np.maximum(a[:,None,:2],b[None,:,:2]);lower=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    area=np.maximum(lower-upper,0).prod(2)
    union=a[:,2:].prod(1)[:,None]+b[:,2:].prod(1)[None,:]-area
    return np.divide(area,union,out=np.zeros_like(area),where=union>0)


def certify_queries(rows,metadata,gt_by_view,scene):
    grouped=defaultdict(list)
    for row in rows:grouped[row['image_id']].append(row)
    result={}
    gt_frames=defaultdict(list)
    for view,camera_rows in gt_by_view.items():
        for row in camera_rows:
            if len(row)<7 or row[6]!=0:gt_frames[(view,int(row[0]))].append(row)
    for image,meta in metadata.items():
        view=f"{scene}_View{meta['view_id']}";frame=meta['frame_id']
        raw=gt_frames[(view,frame)]
        duplicates=Counter(int(r[1]) for r in raw)
        pred=grouped[image];overlap=iou_matrix([r['bbox'] for r in pred],[r[2:6] for r in raw])
        assignment=dict(zip(*linear_sum_assignment(-overlap))) if overlap.size else {}
        for index,row in enumerate(pred):
            target=assignment.get(index);person=None
            if target is not None and overlap[index,target]>=.5:
                plausible=np.flatnonzero(overlap[index]>=.5)
                if len(plausible)==1 and duplicates[int(raw[target][1])]==1 and raw[target][1]>0:
                    person=int(raw[target][1])
            result[(image,index)]=person
    return result,grouped


def owner_states(rows,metadata,labels):
    images=defaultdict(list)
    for row in rows:images[row['image_id']].append(row)
    frames=defaultdict(list)
    for image,meta in metadata.items():
        for index,row in enumerate(images[image]):
            frames[meta['frame_id']].append((image,index,row,meta))
    counts=defaultdict(Counter);totals=Counter();owners={};output={};mixed=defaultdict(set)
    for frame,observations in sorted(frames.items()):
        # Owners use strictly earlier synchronized groups; no same-group GT
        # can certify another camera's current observation.
        for image,index,row,meta in observations:
            key=(image,index);identity=int(row['track_id']);person=labels[key]
            owner=owners.get(identity)
            state='GT_UNKNOWN' if person is None else ('UNANCHORED' if owner is None else ('CORRECT' if owner==person else 'WRONG'))
            output[key]=dict(frame=frame,camera=meta['view_id'],person=person,identity=identity,owner=owner,state=state)
        for image,index,row,meta in observations:
            identity=int(row['track_id']);person=labels[(image,index)];totals[identity]+=1
            if person is not None:counts[identity][person]+=1;mixed[identity].add(person)
        for identity,counter in counts.items():
            if identity in owners:continue
            known=sum(counter.values());person,dominant=counter.most_common(1)[0]
            if known>=3 and known/totals[identity]>=.8 and dominant/known>=.8:owners[identity]=person
    return output,dict(native_global_IDs=len(totals),fixed_causal_owners=len(owners),
        mixed_native_IDs=sum(len(v)>1 for v in mixed.values()),states=dict(Counter(r['state'] for r in output.values())))


def episodes(video,states):
    by_person=defaultdict(list)
    for row in states.values():
        if row['person'] is not None:by_person[(row['camera'],row['person'])].append(row)
    output=[]
    for (camera,person),rows in by_person.items():
        rows.sort(key=lambda r:r['frame']);current=None
        for row in rows:
            continuous=current is not None and row['frame']==current['end_frame']+1
            same=continuous and row['state']=='WRONG' and row['identity']==current['identity'] and row['owner']==current['wrong_owner']
            if current is not None and not same:
                current['end_reason']='GAP_CENSORED' if not continuous else ('NEXT_CERTIFIED_CORRECT' if row['state']=='CORRECT' else 'IDENTITY_OR_CERTIFICATION_CHANGED')
                output.append(current);current=None
            if row['state']=='WRONG':
                if current is None:current=dict(video=video,camera=camera,person=person,identity=row['identity'],wrong_owner=row['owner'],start_frame=row['frame'],end_frame=row['frame'],observed_wrong_camera_frames=1)
                else:current['end_frame']=row['frame'];current['observed_wrong_camera_frames']+=1
        if current is not None:current['end_reason']='VIDEO_OR_OBSERVATION_END_CENSORED';output.append(current)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-runs',type=Path,required=True);p.add_argument('--candidate-runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    original=collect(json.loads(a.original_runs.read_text()));candidate=collect(json.loads(a.candidate_runs.read_text()))
    if set(original)!=set(candidate):raise ValueError('Cannot compare different video scopes')
    data=train_data();results={};all_episodes=[];transitions=Counter();groups=defaultdict(Counter)
    for video,base in original.items():
        other=candidate[video];metadata={r['image_id']:r for r in video_records(data,video)}
        rows={}
        for arm,spec in [('original',base),('candidate',other)]:
            pp=Path(spec['run']['predictions']['raw']['path']);rows[arm]=json.loads(pp.read_text())
        labels,grouped=certify_queries(rows['original'],metadata,base['gt'],base['scene'])
        paired=defaultdict(list)
        for row in rows['candidate']:paired[row['image_id']].append(row)
        if any([(r['bbox'],r['score'],r['category_id']) for r in grouped[image]]!=[(r['bbox'],r['score'],r['category_id']) for r in paired[image]] for image in metadata):
            raise ValueError('Perception/physical query rows differ across policies')
        by_arm={};per=Counter()
        for arm in rows:
            states,summary=owner_states(rows[arm],metadata,labels);by_arm[arm]=states
            episodes_arm=episodes(video,states)
            all_episodes.extend(dict(method=arm,**e) for e in episodes_arm)
            summary.update(error_episode_count=len(episodes_arm),longest_observed_wrong_camera_frames=max([e['observed_wrong_camera_frames'] for e in episodes_arm]+[0]))
            results.setdefault(str(video),{})[arm]=summary
        for key,label in labels.items():
            before=by_arm['original'][key]['state'];after=by_arm['candidate'][key]['state'];per[(before,after)]+=1
            if label is not None:groups[(video,label)][(before,after)]+=1
        transitions.update(per);results[str(video)]['transitions']={f'{b}->{c}':n for (b,c),n in per.items()}
    a.out.mkdir(parents=True)
    fields=['method','video','camera','person','identity','wrong_owner','start_frame','end_frame','observed_wrong_camera_frames','end_reason']
    with (a.out/'IDENTITY_ERROR_EPISODES.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_episodes)
    report=dict(status='COMPLETE_OFFLINE_CAUSAL_OWNER_DIAGNOSTIC',per_video=results,
        certified_wrong_to_correct=transitions[('WRONG','CORRECT')],certified_correct_to_wrong=transitions[('CORRECT','WRONG')],
        transitions={f'{b}->{c}':n for (b,c),n in transitions.items()},
        repair_person_dependence_groups=sum(c[('WRONG','CORRECT')]>0 for c in groups.values()),
        harmed_person_dependence_groups=sum(c[('CORRECT','WRONG')]>0 for c in groups.values()),
        per_person_dependence_groups=[dict(video=v,person=person,transitions={f'{b}->{c}':n for (b,c),n in counts.items()}) for (v,person),counts in sorted(groups.items())],
        criterion='owner fixed after >=3 known past observations, >=80% known coverage and >=80% dominant known person; same-group GT excluded; never future relabelled',
        GT_query_rule='IoU>=0.5 Hungarian, unique plausible GT and nonduplicated event identity, zero-marked ignored; other queries UNKNOWN',
        time_unit='observed camera-frame indices, not physical seconds; gaps/end censored; identity changes split episodes',
        caveat='Dependencies share person/video. UNANCHORED/GT_UNKNOWN are not repairs; always-new fragmentation may hide owner errors, native ID counts and coverage reported separately.',
        input_manifests={str(p):sha256(p) for p in (a.original_runs,a.candidate_runs)},
        episodes_sha256=sha256(a.out/'IDENTITY_ERROR_EPISODES.csv'),GT_in_deployment=False)
    save_json(a.out/'RESULT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_video','input_manifests','per_person_dependence_groups')}),flush=True)


if __name__=='__main__':main()
