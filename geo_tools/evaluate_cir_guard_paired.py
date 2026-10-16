"""Separate offline GT join of all sealed bounded native intervention futures.

Not imported by inference. Overlapping observations are never independent wins.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
from common import DATA,train_data,video_records,save_json,sha256
from cir_evaluation import read_mot
from audit_cir_identity_propagation import certify_queries,owner_states,episodes


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if sha256(a.events)!='106f5e7efce7b0ed08bb1f327d8aa30696ef65dbb5b29fadbdaeec88a44edc74':raise ValueError('Sealed GT-free target drift')
    with gzip.open(a.events,'rt') as stream:
        expected={tuple(r['key']) for line in stream if (r:=json.loads(line))['evidence_state']=='SUSPECT'}
    records=[];seen=set();commits=set();prefixes={};bindings=[]
    for scene in (12,20,21,24):
        path=a.runs/f'scene{scene}/manifest.json';manifest=json.loads(path.read_text())
        if manifest['status']!='COMPLETE_ALL_PREDECLARED_SCENE_TARGETS' or manifest['source_tree_dirty']:
            raise ValueError('All predeclared clean native scenes must finish before GT join')
        commits.add(manifest['source_commit']);bindings.append(dict(path=str(path),sha256=sha256(path)))
        prefix=manifest['prefix']
        if sha256(prefix['path'])!=prefix['sha256']:raise ValueError('Original prefix trajectory drift')
        prefixes[scene]=json.loads(Path(prefix['path']).read_text())
        for spec in manifest['completed']:
            if sha256(spec['result'])!=spec['sha256']:raise ValueError('Case policy result drift')
            case=json.loads(Path(spec['result']).read_text());key=tuple(case['target']['key'])
            if key in seen or key[0]!=scene:raise ValueError('Duplicated or foreign physical intervention')
            seen.add(key)
            if case['status']!='COMPLETE_PAIRED_NATIVE_FUTURE' or not case['identical_target_input']:
                raise ValueError('Unqualified actual paired inputs')
            arms={}
            for arm,branch in case['branches'].items():
                pred=branch['predictions']
                if sha256(pred['path'])!=pred['sha256']:raise ValueError('Paired native trajectory drift')
                arms[arm]=json.loads(Path(pred['path']).read_text())
            records.append((case,arms))
    if seen!=expected or len(seen)!=26 or len(commits)!=1:raise ValueError('Missing/extra cases or mixed sources')
    a.out.mkdir(parents=True)
    save_json(a.out/'POLICY_SEAL.json',dict(status='ALL26_POLICY_OUTPUTS_VERIFIED_BEFORE_GT_JOIN',manifests=bindings,
        source_commit=next(iter(commits)),events_sha256=sha256(a.events)))
    # Only after all policy artifacts are checked and sealed is offline GT read.
    data=train_data();metadata={scene:{r['image_id']:r for r in video_records(data,scene)} for scene in (12,20,21,24)}
    names={v['id']:v['file_name'] for v in data['videos']};gt={};gtsha={}
    for scene,index in metadata.items():
        gt[scene]={}
        for camera in sorted({r['view_id'] for r in index.values()}):
            name=f'{names[scene]}_View{camera}';path=DATA/'train'/name/'gt/gt.txt'
            gt[scene][name]=read_mot(path);gtsha[str(path)]=sha256(path)
    results=[];repaired=set();harmed=set();case_effects=Counter()
    for case,arms in records:
        key=tuple(case['target']['key']);scene=key[0];frame=case['first_frame'];last=case['last_frame']
        index={image:row for image,row in metadata[scene].items() if row['frame_id']<=last}
        prefix=[r for r in prefixes[scene] if index.get(r['image_id'],{}).get('frame_id',10**9)<frame]
        if len(prefix)!=case['prefix_predictions_before_target']:raise ValueError('Original starting prefix mismatch')
        complete={arm:prefix+rows for arm,rows in arms.items()}
        before=complete['C1_KEEP'];after=complete['C1_SINGLE_TARGET_GUARD']
        signature=lambda rows:[(r['image_id'],r['bbox'],r['score'],r['category_id']) for r in rows]
        if signature(before)!=signature(after):raise ValueError('Physical detections differ across intervention arms')
        labels,grouped=certify_queries(before,index,gt[scene],names[scene]);states={}
        for arm,rows in complete.items():states[arm]=owner_states(rows,index,labels)[0]
        input_rows=case['branches']['C1_KEEP']['inputs'][0]['native_scores']
        # Postprocessing can remove empty boxes. Only certify the native target
        # row by public index if count equality proves that none were removed.
        target_mapping=len(grouped[key[1]])==len(input_rows)
        target_person=labels.get((key[1],key[2])) if target_mapping else None
        per_horizon={}
        for horizon in (32,64):
            end=min(last,frame+horizon);counts=Counter();groups=defaultdict(Counter);summaries={}
            for source,person in labels.items():
                f=index[source[0]]['frame_id']
                if not frame<=f<=end:continue
                b=states['C1_KEEP'][source]['state'];c=states['C1_SINGLE_TARGET_GUARD'][source]['state']
                counts[(b,c)]+=1
                if person is not None:groups[(scene,person)][(b,c)]+=1
            for arm in states:
                cropped={k:v for k,v in states[arm].items() if frame<=v['frame']<=end}
                ep=episodes(scene,cropped)
                summaries[arm]=dict(observed_states=dict(Counter(v['state'] for v in cropped.values())),
                    wrong_episodes=len(ep),longest_observed_wrong_camera_frames=max([e['observed_wrong_camera_frames'] for e in ep]+[0]),
                    native_IDs=len({v['identity'] for v in cropped.values()}),episodes=ep)
            per_horizon[str(horizon)]=dict(transitions={f'{b}->{c}':n for (b,c),n in counts.items()},
                wrong_to_correct_observations=counts[('WRONG','CORRECT')],correct_to_wrong_observations=counts[('CORRECT','WRONG')],
                groups=[dict(scene=s,person=p,transitions={f'{b}->{c}':n for (b,c),n in cc.items()}) for (s,p),cc in sorted(groups.items())],
                arms=summaries,end_censored=end<frame+horizon,
                unit='observed camera-frame indices; left-window clipping and end/gaps censored, not seconds')
            if horizon==64:
                repaired.update(g for g,cc in groups.items() if cc[('WRONG','CORRECT')]>0)
                harmed.update(g for g,cc in groups.items() if cc[('CORRECT','WRONG')]>0)
                repair=counts[('WRONG','CORRECT')];harm=counts[('CORRECT','WRONG')]
                effect='HAS_CERTIFIED_REPAIR_WITHOUT_CERTIFIED_HARM' if repair and not harm else 'HAS_CERTIFIED_HARM' if harm else 'NO_CERTIFIED_CHANGE'
                case_effects[effect]+=1
        results.append(dict(target=key,target_GT_person=target_person,target_native_to_public_row_proven=target_mapping,
            veto_edges=case['branches']['C1_SINGLE_TARGET_GUARD']['veto_edges'],
            guard_joint_solves=case['branches']['C1_SINGLE_TARGET_GUARD']['guard_joint_solves'],horizons=per_horizon))
    report=dict(status='COMPLETE_ALL26_PAIRED_NATIVE_OFFLINE_DIAGNOSTIC',cases=results,
        cases_with_veto=sum(r['veto_edges']>0 for r in results),case_effects=dict(case_effects),
        unique_repaired_scene_person_groups=sorted(repaired),unique_harmed_scene_person_groups=sorted(harmed),
        dependence='Overlapping futures are dependent. Do not pool repeated camera-frame observations as independent repairs or trials.',
        owner_metric='Past-only anchored native identity consistency, not optimal IDF1 matching. UNKNOWN/unanchored are not certified repairs.',
        full_video_quality_qualified=False,scientific_success=False,GT_in_inference=False,raw_GT_modified=False,
        policy_seal_sha256=sha256(a.out/'POLICY_SEAL.json'),raw_GT_sha256=gtsha)
    save_json(a.out/'RESULT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','raw_GT_sha256')}),flush=True)


if __name__=='__main__':main()
