"""Audit scientific artifacts and exact immutable sources without opening heldout data."""
import collections
from jev_phase12_common import *

def main():
    protect()
    from jev_phase10_common import FOUNDATION,CACHE,B2
    old=json.loads((PREVIOUS/'native_capture_v1/video12/compat/START_MANIFEST.json').read_text())['binding']
    foundation={k:{'path':str(p),'SHA256':sha(p)} for k,p in [('foundation',FOUNDATION),('cache_index',CACHE/'index.jsonl'),('B2',B2)]}
    for k,r in foundation.items():assert r['SHA256']==old[k+'_SHA256']
    assert sha(OUT/'visual_dataset_v1/DATASET.pth')==json.loads((REPORTS/'MATCH_OFFLINE_VALIDATION.json').read_text())['dataset_SHA256']
    protocol=json.loads((REPORTS/'ONLINE_PROTOCOL.json').read_text());online=json.loads((REPORTS/'MATCH_VALIDATION_RESULTS.json').read_text());joint=json.loads((REPORTS/'GLOBAL_MCMOT_VALIDATION.json').read_text())
    assert online['status']=='COMPLETE' and online['native_video_actors']==168 and len(online['cases'])==len(joint['cases'])==56
    assert {r['case']['name'] for r in online['cases']}=={r['case']['name'] for r in joint['cases']}=={r['name'] for r in protocol['cases']}
    main_source=OUT/'source_online_v3';accepted={subprocess.check_output(['git','rev-parse','HEAD'],cwd=main_source,text=True).strip(),subprocess.check_output(['git','rev-parse','HEAD'],cwd=OUT/'source_online_v2',text=True).strip()}
    for name,digest in protocol['scripts_SHA256'].items():assert sha(main_source/name)==digest,name
    sources=collections.Counter();actors=[];metric_files=0;detection_deltas=[]
    for case in online['cases']:
        name=case['case']['name'];global_case=next(r for r in joint['cases'] if r['case']['name']==name)
        assert case['status']=='COMPLETE'
        for scope in ['strict_online','canonical_GMT_filtered']:
            a=case['pooled_metric_artifacts'][scope];assert sha(a['combined_metrics_path'])==a['SHA256'];metric_files+=1
            detection_deltas.append({'case':name,'scope':scope,'joint_minus_per_camera_DetA':global_case['metrics'][scope]['metrics']['DetA']-case['pooled_metrics'][scope]['DetA']})
            g=global_case['metrics'][scope];gp=Path(g['manifest']['trackeval_gt']).parents[1]/'evaluation/metrics.json';assert sha(gp)==g['combined_metrics_SHA256'];metric_files+=1
        for video in VAL:
            p=OUT/'validation_closed_loop_v1'/name/f'video{video:02d}/RESULT.json';r=json.loads(p.read_text())
            assert r['status']=='COMPLETE' and r['video']==video and r['case']==case['case'] and r['actual_mutated_state_online']
            assert not r['binding']['dirty'] and not r['Full24'] and not r['official_TEST'] and r['heldout']=='SEALED'
            commit=r['binding']['source_commit'];assert commit in accepted
            if commit!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=main_source,text=True).strip():assert name in ['GMT_OFF','Fixed'] or case['case']['model'] in ['CandidateMLP','CandidateDeepSets','CandidateJEV']
            for n in ['strict_predictions','canonical_predictions']:assert sha(r[n]['path'])==r[n]['SHA256']
            assert r['paired_H32_causal_audit']['actual_current_and_future_matches_complete_online_path']
            sources[commit]+=1;actors.append({'case':name,'video':video,'path':str(p),'SHA256':sha(p)})
    fits=[];weights=0
    for p in sorted((OUT/'formal_training_v1').glob('*/joint/seed*/RESULT.json')):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE_STABLE' and not r['binding']['dirty'] and r['actual_updates']==900
        for c in r['checkpoints'].values():assert sha(c['path'])==c['SHA256'];weights+=1
        fits.append({'model':r['model'],'seed':r['seed'],'result_SHA256':sha(p)})
    assert len(fits)==51
    for name in ['NATIVE_PARITY','NATIVE_DEPLOYMENT_REGRESSION','STALE_TYPED_INPUT_CONTRACT','FULL_LIFECYCLE_GUARD']:assert json.loads((REPORTS/(name+'.json')).read_text())['status']=='PASS'
    sealed_outputs=[str(p) for p in OUT.glob('**/video*') if p.name in ['video20','video21','video22']];assert not sealed_outputs
    modified=subprocess.check_output(['git','diff',BASE,'--name-only','--','reports/JEV_PHASE5','reports/JEV_PHASE6','reports/JEV_PHASE7','reports/JEV_PHASE8','reports/JEV_PHASE9','reports/JEV_PHASE10'],cwd=ROOT,text=True).strip();assert not modified
    save(REPORTS/'DELIVERY_INTEGRITY.json',{'status':'PASS','foundation_cache_B2_unchanged':foundation,'frozen_main_actor_source':str(main_source),'frozen_primary_protocol_scripts_exact':True,'actor_source_counts':dict(sources),'complete_actor_count':len(actors),'prediction_SHA_checks':336,'metric_SHA_checks':metric_files,'DetA_scope_comparison':{'values':detection_deltas,'reason':'same geometry need not imply identical DetA: TrackEval/trackeval/metrics/hota.py multiplies similarity by global_alignment_score before Hungarian; joining cameras changes this identity-dependent matching score. These are different metric scopes, not substitute predictions.'},'formal_fit_count':len(fits),'checkpoint_SHA_checks':weights,'fits':fits,'actors':actors,'old_report_changes':[],'heldout_output_dirs':sealed_outputs,'heldout':'SEALED','Full24':False,'official_TEST':False,'scope':'artifact integrity verification; no new training, metric selection, heldout read or repeated video inference'})
    print('PHASE12_DELIVERY_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
