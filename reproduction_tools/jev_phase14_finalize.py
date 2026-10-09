"""Aggregate completed immutable evidence; never choose or modify a controller."""
import argparse,collections,gzip,time
import numpy as np
from jev_phase14_common import *

PHASES=['formal','onpolicy','oldloss_onpolicy','offpolicy']
KEYS=['HOTA','AssA','IDF1','IDSW','MOTA','Frag']
def ref(path):return dict(path=str(path),SHA256=sha(path))
def stats(rows,keys):
    return {k:dict(mean=float(np.mean([r[k] for r in rows])),std_seed_population=float(np.std([r[k] for r in rows])),seeds=[r[k] for r in rows]) for k in keys}
def counts(rows):
    c=collections.Counter()
    for r in rows:c.update(r)
    return dict(c)
def trace(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]

def main(wait=False):
    protect();source=binding();p=read(REPORTS/'FORMAL_PROTOCOL.json');variants=p['architectures'];seeds=p['seeds']
    queues=[OUT/'completion_queue_v2/RESULT.json',OUT/'metrics_queue_v2/RESULT.json',OUT/'diagnostic_queue_v1/RESULT.json',OUT/'external_eval_v1/completion_queue_v1/RESULT.json']
    while any(not q.exists() for q in queues):
        assert wait,'complete prerequisites required before final report'
        time.sleep(10)
    queue_results=[read(q) for q in queues];assert all(q['status']=='COMPLETE' for q in queue_results),[(str(f),q['status'],q.get('failed')) for f,q in zip(queues,queue_results)]
    online=[];official=[];training=[];risk=[];summaries={};matlab_summaries={}
    for phase in PHASES:
        summaries[phase]={};matlab_summaries[phase]={}
        for variant in variants:
            group=[];matgroup=[]
            for seed in seeds:
                name=f'{variant}_seed{seed}';folder=OUT/f'{phase}_pooled_v2'/name;r=read(folder/'RESULT.json');t=read(folder/'TAXONOMY.json');assert r['status']==t['status']=='COMPLETE'
                m=r['strict_pooled_TrackEval'];group.append(m)
                video_results=[]
                for v in DEV:
                    f=OUT/f'{phase}_online_v2'/name/f'video{v:02d}/RESULT.json';d=read(f);assert d['status']=='COMPLETE' and d['actual_mutated_state_online'] and d['GTA_throw_mock']
                    assert d['frames'] in [1200,1029,1052] and d['counts']['payloads']==2*d['frames']-1
                    video_results.append(dict(video=v,frames=d['frames'],metrics=d['strict_online_metrics'],identity_summary=d['identity_summary'],decision_summary=d['decision_summary'],result=ref(f),source_commit=d['binding']['source_commit'],final_complete_native_state_SHA256=d['final_complete_native_state_SHA256']))
                online.append(dict(phase=phase,variant=variant,seed=seed,actual_updates=20000 if phase=='formal' else 24000,metrics=m,
                    canonical_future_filtered_separate=r['canonical_pooled_TrackEval'],taxonomy_counts=t['counts'],videos=video_results,pooled=ref(folder/'RESULT.json'),taxonomy=ref(folder/'TAXONOMY.json'),source_commit=r['binding']['source_commit']))
                f=OUT/f'{phase}_official_matlab_v2'/name/'RESULT.json';d=read(f);assert d['status']=='COMPLETE' and d['unchanged_official_metric_functions'];raw=d['reports']['raw_predictions'];matgroup.append(raw)
                official.append(dict(phase=phase,variant=variant,seed=seed,CVIDF1=raw['CVIDF1'],CVMA=raw['CVMA'],Identity_counts=raw['Identity_counts'],interleaved_CLEAR_counts=raw['interleaved_CLEAR_counts'],
                    canonical_separate={k:d['reports']['canonical_predictions'][k] for k in ['CVIDF1','CVMA']},native_version=d['native_version'],result=ref(f),native_report=raw['native_report'],source_commit=d['binding']['source_commit']))
                f=OUT/f'{phase}_v2'/variant/'availability_joint'/f'seed{seed}/RESULT.json';d=read(f);assert d['status']=='COMPLETE' and d['actual_updates']==(20000 if phase=='formal' else 24000);assert sha(d['checkpoint']['path'])==d['checkpoint']['SHA256']
                training.append(dict(phase=phase,variant=variant,seed=seed,actual_updates=d['actual_updates'],objective=d.get('training_objective',d['loss']),checkpoint=d['checkpoint'],source_commit=d['binding']['source_commit'],result=ref(f),reserved_TRAIN_input_metrics=d['final'],own20k_state_reserved_input_metrics=d.get('own_reserved_final'),own_state_label_support=d.get('own_state_label_support'),sampler_groups=d.get('sampler_groups'),initial=d.get('initial')))
                f=OUT/'matched_TRAIN_native_risk_v1'/phase/name/'RESULT.json';d=read(f);assert d['status']=='COMPLETE' and len(d['results'])==4
                c=counts([x['audit']['counts'] for x in d['results']]);risk.append(dict(phase=phase,variant=variant,seed=seed,counts=c,normal_retention=c.get('normal_correct_retained',0)/max(1,c.get('normal_supported',0)),result=ref(f),videos=[dict(video=x['video'],audit=x['audit'],trace=x['trace']) for x in d['results']]))
            summaries[phase][variant]=stats(group,KEYS);matlab_summaries[phase][variant]=stats(matgroup,['CVIDF1','CVMA'])
    common=dict(status='COMPLETE',binding=source,frozen_evidence=ref(REPORTS/'PHASE13_FROZEN_EVIDENCE.json'),preregistered=ref(REPORTS/'PREREGISTRATION.json'),official_TEST=False,Full24=False,heldout='SEALED')
    save(REPORTS/'ONLINE_VALIDATION.json',dict(**common,cases=online,seed_summary=summaries,scope='36 exact six-camera pooled cases from108 complete native videos; one pooled score perseed, raw primary, canonical future-filtering separately; frozen frontend already exposed all24TRAIN',execution_queue=ref(queues[0])))
    save(REPORTS/'OFFICIAL_MATLAB_RESULTS.json',dict(**common,cases=official,seed_summary=matlab_summaries,scope='actual unchanged official MATLAB metric code; sum per-scene raw counts; sequential CVIDF1, interleaved CVMA, no mean-of-video scores',queue=ref(queues[1])))
    contrasts=[]
    for phase in PHASES:
        for comparator in ['fixed_question','set_transformer']:
            a=[r for r in online if r['phase']==phase and r['variant']=='multi_question'];b=[r for r in online if r['phase']==phase and r['variant']==comparator]
            contrasts.append(dict(phase=phase,comparator=comparator,seed_deltas=[dict(seed=x['seed'],metrics={k:x['metrics'][k]-y['metrics'][k] for k in KEYS},taxonomy={k:x['taxonomy_counts'].get(k,0)-y['taxonomy_counts'].get(k,0) for k in ['FALSE_MERGE','FALSE_SPLIT','FALSE_BIRTH']}) for x,y in zip(a,b)],
                mean_deltas={k:summaries[phase]['multi_question'][k]['mean']-summaries[phase][comparator][k]['mean'] for k in KEYS}))
    save(REPORTS/'FAIR_BASELINE_RESULTS.json',dict(**common,architectures=variants,parameters=p['parameters'],common_input_tasks=['WHO','AVAILABILITY','TRUST'],same_labels_options_solver_updates=True,training=training,contrasts=contrasts,formal_summary=summaries['formal'],
        pilot=ref(REPORTS/'PILOT_V2_RESULTS.json'),excluded_models='MLP/MOTIP do have bounded equal-input pilots; absence head did not meet pilot qualification, no20k architecture superiority inferred from1000update failure',historical_controls='PhaseXIII Full/Fixed/Set20k are separate supervision/temperature controls, not evidence-equal new typed-loss baselines',statistical_scope='three seeds, exploratory preregistered effect gate; no statistical significance claim'))
    save(REPORTS/'ON_POLICY_STABILITY.json',dict(**common,training=[r for r in training if r['phase']!='formal'],TRAIN_native_risk=risk,development_summary=summaries,development_taxonomy=[dict(phase=r['phase'],variant=r['variant'],seed=r['seed'],counts=r['taxonomy_counts']) for r in online],
        same_initial20k_same_extra4000=True,arms=['new loss own-state','old loss same own-state','new loss original off-policy'],same_deployment_availability_mapping=True,no_extra4k_architecture_attribution=True,
        scope='all144 actual mutated-state first256 TRAIN rollouts; GT offline after logits; observed wrong-anchor/newmix/birth/normal continuation; reserved input probes are distinct from actual new native trajectories',protocol=ref(REPORTS/'ON_POLICY_PROTOCOL.json'),diagnostic_queue=ref(queues[2])))
    ab=read(REPORTS/'STRUCTURED_ACTION_LOSS_ABLATION.json');ab.update(status='COMPLETE_BOUNDED_PILOT_AND_MATCHED_EXTRA4K',formal_equal_supervision=ref(REPORTS/'FAIR_BASELINE_RESULTS.json'),matched_objective_controls=ref(REPORTS/'ON_POLICY_STABILITY.json'),cost_semantics=ref(REPORTS/'ACTION_COST_SEMANTICS.json'),formal_all_four_losses_grid='NOT_RUN: user explicitly requests qualified bounded expansion, not unselective model×loss×seed20k sweep')
    save(REPORTS/'STRUCTURED_ACTION_LOSS_ABLATION.json',ab)
    # gzip container timestamps differ; compare actual decompressed journal bytes.
    parity=[]
    for v in variants:
        for video in DEV:
            a=OUT/'formal_online_v2'/f'{v}_seed20261009'/f'video{video:02d}';b=OUT/'formal_online_v2'/f'{v}_seed20261009_uncached'/f'video{video:02d}';ra=read(a/'RESULT.json');rb=read(b/'RESULT.json')
            equal={k:ra[k]==rb[k] for k in ['strict_online_metrics','canonical_future_filtered_metrics','final_complete_native_state_SHA256']}
            equal['raw_prediction_SHA']=ra['raw_predictions']['SHA256']==rb['raw_predictions']['SHA256']
            equal['canonical_prediction_SHA']=ra['canonical_predictions']['SHA256']==rb['canonical_predictions']['SHA256']
            for field in ['COMMITS','QUESTIONS']:
                with gzip.open(a/(field+'.jsonl.gz'),'rb') as h:g=h.read()
                with gzip.open(b/(field+'.jsonl.gz'),'rb') as h:f=h.read()
                equal[field+'_decompressed_bytes']=g==f
            assert all(equal.values()),(v,video,equal)
            parity.append(dict(variant=v,seed=20261009,video=video,frames=ra['frames'],equal=equal,cached=ref(a/'RESULT.json'),uncached=ref(b/'RESULT.json'),complete_native_state_SHA256=ra['final_complete_native_state_SHA256'],numeric_logit_difference=0.))
    save(REPORTS/'FULL_VIDEO_CACHE_PARITY.json',dict(**common,cases=parity,status_detail='PASS',scope='nine independent complete videos with actual policy-mutated state; exact raw/canonical output, every serialized NN logit/legal query, native commit and full final state; gzip header dates excluded',engineering_resume_recycling=ref(REPORTS/'NATIVE_CACHE_PARITY.json'),invalidation=ref(REPORTS/'CACHE_INVALIDATION_TESTS.json')))
    paired=[];paired_summary=[]
    for seed in seeds:
        path=OUT/'paired_native_future_v1'/f'seed{seed}/RESULT.json';d=read(path);assert d['status']=='COMPLETE' and len(d['results'])==156
        bases={(x['prefix']['video'],x['prefix']['frame']):x for x in d['results'] if x['condition']=='unchanged'}
        for x in d['results']:
            baseline=bases[x['prefix']['video'],x['prefix']['frame']];assert x['starting_state_SHA256']==baseline['starting_state_SHA256']
            aa=trace(x['trace']['path']);bb=trace(baseline['trace']['path']);assert [(r['key'],r['row']) for r in aa]==[(r['key'],r['row']) for r in bb]
            diff=sum(a['identity']!=b['identity'] for a,b in zip(aa,bb));c=x['audit']['counts'];bc=baseline['audit']['counts']
            paired.append(dict(seed=seed,video=x['prefix']['video'],frame=x['prefix']['frame'],condition=x['condition'],starting_state_SHA256=x['starting_state_SHA256'],counts=c,delta_counts={k:c.get(k,0)-bc.get(k,0) for k in set(c)|set(bc)},mutated_future_changed_committed_rows=diff,trace=x['trace'],checkpoint=x['checkpoint'],prefix=x['prefix']))
    for condition in read(REPORTS/'PAIRED_NATIVE_PROTOCOL.json')['factors']:
        group=[x for x in paired if x['condition']==condition];paired_summary.append(dict(condition=condition,prefix_seed_cases=len(group),counts=counts([x['counts'] for x in group]),delta_counts=counts([x['delta_counts'] for x in group]),changed_committed_rows=sum(x['mutated_future_changed_committed_rows'] for x in group)))
    save(REPORTS/'PAIRED_NATIVE_FUTURE_RESULTS.json',dict(**common,cases=paired,summary=paired_summary,scope='468 actual same-prefix16scene-frame mutated-future branches, TRAIN only, historical Full20k three seeds, OOD inference dependency only',protocol=ref(REPORTS/'PAIRED_NATIVE_PROTOCOL.json')))
    for name in ['IDENTITY_MEMORY_FACTORIAL','CROSS_CAMERA_EVIDENCE_AUDIT']:
        d=read(REPORTS/(name+'.json'));d.update(status='COMPLETE_FROZEN_INPUT_AND_PAIRED_NATIVE_DEPENDENCY',paired_actual_mutated_future=ref(REPORTS/'PAIRED_NATIVE_FUTURE_RESULTS.json'),actual_official_cross_camera=ref(REPORTS/'OFFICIAL_MATLAB_RESULTS.json'),retrained_structural_claim=False);save(REPORTS/(name+'.json'),d)
    external=[]
    for item in queue_results[3]['done']:
        f=Path(item['result']);d=read(f);assert d['status']=='COMPLETE' and d['actual_mutated_state_online'] and d['GT_actor_inputs'] is False
        external.append(dict(variant=d['variant'],seed=d['seed'],historical=d['historical'],metrics=d['metrics'],clipped_GT_sensitivity=d['clipped_GT_sensitivity'],official_MATLAB=d['official_MATLAB'],trained=d['trained'],temperature=d['temperature'],result=ref(f),source_commit=d['binding']['source_commit']))
    assert len(external)==19
    ex_summary={}
    for hist in [True,False]:
        for v in ['cosine','full','fixed_question','multi_question','set_transformer']:
            g=[d for d in external if d['variant']==v and d['historical']==hist]
            if g:ex_summary[('Phase13_' if hist else 'Phase14_')+v]=stats([x['metrics'] for x in g],KEYS)
    save(REPORTS/'EXTERNAL_GENERALIZATION_RESULTS.json',dict(**common,cases=external,seed_summary=ex_summary,known_training_list_unseen_dataset_transfer=True,absolute_pretraining_exposure_unverified=True,official_full7camera_benchmark_claim=False,scope='fixed C1/C2 first320 real observed2FPS samples,160s; original hyperparameters transferred without tuning or resampling; old/new20k controllers and cosine, raw projected GT primary with clip sensitivity separate',dataset=ref(OUT/'external_eval_v1/DATA_MANIFEST.json'),frontend=ref(OUT/'external_eval_v1/stage1_cache_v1/RESULT.json'),exposure=ref(REPORTS/'PRETRAIN_EXPOSURE_AUDIT.json'),queue=ref(queues[3])))
    source_paths=[OUT/x for x in ['source_pilot_v2','source_engineering_v2','source_pipeline_v2','source_latency_v1','source_external_v1','source_diagnostic_v1']];interfaces=[]
    relevant=['configs/VISION_test.yaml','configs/VISION_stage1.yaml','gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/jev_native_state.py','gtr/modeling/jev_perception_cache.py','reproduction_tools/jev_phase13_runtime.py','reproduction_tools/cache_jev_phase13_stage1.py','reproduction_tools/jev_phase13_official_matlab_eval.m']
    for folder in source_paths:
        interfaces.append(dict(path=str(folder),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip(),files={f:sha(folder/f) for f in relevant}))
    save(REPORTS/'SOURCE_INTERFACE_MANIFEST.json',dict(**common,sources=interfaces,all_actual_training_checkpoints_verified=True,all_protected_Phase13_reports_SHA_unchanged=protect(),new_models=training,data_manifest=ref(REPORTS/'QUESTION_SUPERVISION_ELIGIBILITY.json'),Stage1=read(OUT/'external_eval_v1/stage1_cache_v1/RESULT.json')['Stage1']))
    latency=read(REPORTS/'LATENCY_ATTRIBUTION.json');learned=[x for x in latency['cases'] if x['variant']!='cosine']
    efficiency=all(all(t['total_Stage2_ms']['p95']<=10. and t['full_sceneFPS']>=25. for t in x['real_image_trials']) for x in learned)
    formal=summaries['formal'];contrasts_formal=[x for x in contrasts if x['phase']=='formal']
    independent=all(x['mean_deltas']['HOTA']>=.5 and x['mean_deltas']['AssA']>=.5 and all(t['metrics']['HOTA']>=0 for t in x['seed_deltas']) and all(t['taxonomy']['FALSE_MERGE']<=0 and t['taxonomy']['FALSE_SPLIT']<=0 and t['taxonomy']['FALSE_BIRTH']<=0 for t in x['seed_deltas']) for x in contrasts_formal)
    frozen=read(REPORTS/'PHASE13_FROZEN_EVIDENCE.json');old=read(REPORTS/'ERROR_ATTRIBUTION.json')['aggregate_counts']['formal/fixed_question'];stability={}
    for v in variants:
        c=counts([r['taxonomy_counts'] for r in online if r['phase']=='formal' and r['variant']==v]);stability[v]=dict(pass_gate=formal[v]['IDSW']['mean']<=102.5 and formal[v]['HOTA']['mean']>=frozen['Fixed_Question']['HOTA'] and all(c.get(k,0)<=old.get(k,0) for k in ['FALSE_MERGE','FALSE_SPLIT','FALSE_BIRTH']),IDSW_mean=formal[v]['IDSW']['mean'],threshold=102.5,counts=c)
    gates=dict(G0=dict(status='PASS',evidence=['NATIVE_CACHE_PARITY.json','CACHE_INVALIDATION_TESTS.json','FULL_VIDEO_CACHE_PARITY.json']),G1=dict(status='PASS',evidence=['ERROR_ATTRIBUTION.json','UNKNOWN_CHOICE_FORENSICS.json']),
        G2=dict(status='PASS_LEARNABILITY_ONLY',evidence=['QUESTION_SUPERVISION_ELIGIBILITY.json','FAIR_BASELINE_RESULTS.json'],scope='WHO/AVAILABILITY/TRUST; not learned MATCH/REACT/MEM lifecycle'),
        G3=dict(status='PASS_ACTION_SEMANTICS',learned_REACT=False,learned_MEMORY_WRITE=False),G4=dict(status='PASS' if any(x['pass_gate'] for x in stability.values()) else 'FAIL',architectures=stability),
        G5=dict(status='PASS' if independent else 'FAIL',same_evidence_contrasts=contrasts_formal,statistical_significance_claim=False),G6=dict(status='PASS_EVALUATION_COMPLETENESS',evidence='OFFICIAL_MATLAB_RESULTS.json',performance_advantage_claim='requires metric comparisons, no autoPASS'),
        G7=dict(status='PASS' if efficiency else 'FAIL',Stage2_p95_ms_max=10.,sceneFPS_min=25.,historical_G8='FAIL'),G8=dict(status='NO_GO_ABSOLUTE_UNSEEN_FULL_SYSTEM',known_training_list_external_transfer='COMPLETE',clean_Stage1='NOT_RUN_QUALIFICATION_FAILED',initializer_manifest='UNVERIFIED',evidence='PRETRAIN_EXPOSURE_AUDIT.json'))
    save(REPORTS/'FINAL_GO_NO_GO.json',dict(**common,gates=gates,scientific_overall='NO_GO' if not(independent and efficiency and any(x['pass_gate'] for x in stability.values())) else 'GO',execution_deliveries='COMPLETE',
        complete_MATCH_REACT_MEMORY_expansion='NO_GO: real REACT support9, no MEMORY reward labels; Q3 trust is not write/action reward',
        preserve_all_negative_results=True,no_best_checkpoint_selection=True,no_hyperparameter_or_gate_changes=True,
        matrix=dict(pilot_v1_superseded=24,pilot_v2_equal_evidence=24,formal20k=9,extra4k=27,complete_DEV_native_videos=108,full_uncached_parity_videos=9,official_DEV_pooled_cases=36,TRAIN_native_risk_rollouts=144,paired_native_future_branches=468,external_native_cases=19),
        conditional_NOT_RUN={'clean_Stage1':'available exact initializer provenance and dedicated configuration/budget not qualified','learned_REACT_MEMORY':'genuine label qualifications missing','MLP_MOTIP20k':'frozen pilot qualification failed','memory_factor_retraining':'OOD dependency diagnosis only; no independent stability advantage warranting expanded training'}))
    print('PHASE14_ALL_COMPLETED_EVIDENCE_AGGREGATED',gates,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--wait',action='store_true');main(a.parse_args().wait)
