"""Complete-video MATCH dataset with exact offline GT-frame correction.

Only windows that contain a changed detection/GT match require new branch
rollouts. Unaffected windows retain their identical frozen outcomes. The
selection is exhaustive by geometry/coordinate differences, not utility.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
from audit_jev_phase5 import BASE,OUT,TRAIN,CACHE,records,key,sha
from run_segmented_small_gate import atomic,trace_path


def prepare():
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import detection_target,ordered_production_keys
    cache=FrozenPerceptionCache(CACHE);gt=load_gt(TRAIN);shifted={(v,w,f-1):i for (v,w,f),i in gt[1].items()};changed=defaultdict(list);eligible=[];controls=[]
    for v in (6,7):
        for k in sorted(k for k in cache.keys() if k[0]==v):
            payload=cache.load(*k)
            for row,b in enumerate(payload['pred_boxes']):
                args={'video_id':v,'frame':k[1],'view':k[2],'box':b.tolist(),'image_size':payload['image_size'],'gt_by_image':gt[2],'image_meta':gt[3]}
                old=detection_target(images=gt[1],**args);new=detection_target(images=shifted,**args)
                if old!=new:changed[v].append({'key':list(k),'row':row,'legacy_gt':old,'aligned_gt':new})
        changed_frames={r['key'][1] for r in changed[v]};rows=sorted([r for r in records(v) if r['question_type']=='MATCH_DECISION'],key=key)
        affected=[r for r in rows if any(key(r)[0]<=f<=key(r)[0]+8 for f in changed_frames)]
        unaffected=[r for r in rows if not any(key(r)[0]<=f<=key(r)[0]+8 for f in changed_frames)]
        # Predeclared four evenly spaced unaffected control windows per video.
        import numpy as np
        control_rows=[unaffected[i] for i in np.round(np.linspace(0,len(unaffected)-1,min(4,len(unaffected)))).astype(int)]
        eligible.extend((v,key(r)) for r in affected+control_rows);controls.extend((v,key(r)) for r in control_rows)
    source=json.loads((BASE/'plan.json').read_text());grouped=defaultdict(list)
    for v in (6,7):
        ordered,_=ordered_production_keys([k for k in cache.keys() if k[0]==v],lambda k:cache.load(*k));indices={k:i for i,k in enumerate(ordered)}
        for vid,k in eligible:
            if vid!=v:continue
            index=indices[(v,k[0],k[1])];chunk=next(c for c in source['chunks'] if c['video_id']==v and c['key_start']<=index<c['key_end']);grouped[chunk['name']].append(list(k))
    chunks=[]
    for name,keys in sorted(grouped.items()):
        c=next(c for c in source['chunks'] if c['name']==name);chunks.append({**c,'selected_events':keys,'trace':str(trace_path(c['video_id']))})
    plan={'status':'PREDECLARED','task':'match_relabel','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_plan_sha256':sha(BASE/'plan.json'),'horizons':[8],'selected_records':len(eligible),'complete_video_match_records':4303,
          'source_snapshot_commit':source['binding']['source_commit'],'chunks':chunks,'controls':[[v,list(k)] for v,k in controls],
          'all_changed_detection_targets':dict(changed),'full_trace_and_future_maps_retained':True,'max_events_used':False,'official_test_read':False,
          'selection':'all MATCH windows in video06/video07 containing any detection GT-match difference caused by frame alignment; plus 4 evenly spaced unaffected controls/video; no utility/outcome-based filtering',
          'reuse_proof':'Frozen boxes and scores determine GT match independently of branch track IDs. A window with no changed match has identical score inputs and outcomes under the GT correction.',
          'label_protocol':'H8 fixed OFF action-category continuation; branch assignments recomputed; window identity anchors reset as in frozen control',
          'purpose':'minimal MATCH-only training, complete TRAIN video07 / val video06, frozen successful original MATCH checkpoint retained'}
    atomic(OUT/'match_relabel'/'plan.json',plan);print(json.dumps({'selected_records':len(eligible),'chunks':len(chunks),'changed_targets':{v:len(r) for v,r in changed.items()},'controls':len(controls)}))


def combine():
    from jev_dataset_tools import make_record
    from jev_compact_dataset import build_compact_dataset
    plan=json.loads((OUT/'match_relabel'/'plan.json').read_text());new={(r['state']['online_context']['video_id'],key(r)):r for r in (json.loads(l) for l in (OUT/'match_relabel'/'records.jsonl').open())}
    controls={(v,tuple(k)) for v,k in plan['controls']};changed=Counter();best_changed=Counter();total=Counter();paths=[]
    for v in (6,7):
        data=[]
        for r in records(v):
            if r['question_type']!='MATCH_DECISION':continue
            k=(v,key(r));total[v]+=1
            if k in new:
                outcomes=new[k]['aligned_horizon_outcomes']['8'];changed[v]+=outcomes!=r['action_outcomes']
                if k in controls:assert outcomes==r['action_outcomes'],'unaffected-window reuse proof failed'
                replacement=make_record(dataset=r['dataset'],sequence=r['sequence'],frame=r['frame'],view=r['view'],question_type=r['question_type'],state=r['state'],legal_actions=r['legal_actions'],action_outcomes=outcomes,gmt_checkpoint_sha256=r['gmt_checkpoint_sha256'],horizon=8)
                best_changed[v]+=replacement['best_actions']!=r['best_actions'];r=replacement
            r['label_coordinate_contract']='cache frame0 -> annotation frame_id1; view0 -> view_id1'
            r['label_continuation_contract']='frozen OFF action categories, recomputed branch assignments'
            data.append(r)
        path=OUT/'match_training'/f'video{v:02d}_aligned_match.jsonl';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in data));paths.append(path)
    assert dict(total)=={6:2600,7:1703}
    manifest=build_compact_dataset(paths,OUT/'match_training'/'compact')
    split=json.loads((BASE/'small_h8_training_current_head_video06_video07'/'policy_split.json').read_text());split.update(source_files=[str(p) for p in paths],source_sha256={str(p):'sha256:'+sha(p) for p in paths},source_record_count=4303,train_record_count=1703,val_record_count=2600,purpose='phase5_minimal_match_only_coordinate_corrected_training',official_test_used_for_search=False)
    atomic(OUT/'match_training'/'policy_split.json',split)
    summary={'status':'COMPLETE','records_by_video':dict(total),'changed_outcomes_by_video':dict(changed),'changed_best_action_sets_by_video':dict(best_changed),'unaffected_controls_exact':True,'coordinate_fix_applied_to_all_complete_video_match_records':True,'compact_manifest_sha256':sha(OUT/'match_training'/'compact'/'manifest.json'),'official_test_read':False,'legacy_proxy_utility_limitations_retained':True,'training_checkpoint_selection':'fixed last epoch 20, seed20261003; no held-out tuning; val06 temperature only'}
    atomic(OUT/'MATCH_TRAINING_DATASET.json',summary);print(json.dumps(summary))

if __name__=='__main__':globals()[sys.argv[1]]()
