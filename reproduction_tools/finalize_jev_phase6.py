"""Require completed bounded evidence and write honest conditional Phase VI verdicts."""
import json
from pathlib import Path
import subprocess
from jev_phase6_common import OUT,REPORTS,ROOT,B2,B2_SHA,sha,save,protect_anchor

def read(path):return json.loads(Path(path).read_text())

def main():
    protect_anchor();baselines={};heads={};deltas={}
    for video in (24,23):
        records={}
        for condition in ['M0','M1','M2','M3','M4','M5']:
            path=OUT/f'memory_baselines_native/video{video}/{condition}/result.json';r=read(path)
            if r['status']!='PASS' or not r['native_MATCH_transition_contract'] or r['checkpoint_sha256']!=B2_SHA:raise AssertionError('incomplete/unbound baseline')
            if sha(r['result']['predictions'])!=r['prediction_sha256']:raise AssertionError('baseline prediction changed')
            records[condition]={'metrics':r['metrics'],'prediction_sha256':r['prediction_sha256'],'source_result_sha256':sha(path),
                'source':str(path),'delta_vs_M0':{k:r['metrics'][k]-records['M0']['metrics'][k] for k in ['HOTA','AssA','IDF1','IDSW']} if condition!='M0' else {k:0 for k in ['HOTA','AssA','IDF1','IDSW']}}
            if condition=='M2' and r['prediction_sha256']!=records['M0']['prediction_sha256']:raise AssertionError('latest10 contract')
        baselines[str(video)]=records
        path=OUT/f'standalone_closed_loop_native/REACT/video{video}/result.json';heads[str(video)]=read(path)
        if sha(heads[str(video)]['result']['predictions'])!=heads[str(video)]['prediction_sha256']:raise AssertionError('head prediction changed')
        deltas[str(video)]={k:heads[str(video)]['metrics'][k]-records['M0']['metrics'][k] for k in ['HOTA','AssA','IDF1','IDSW']}
    memory_fit=read(OUT/'standalone_heads_native/MEMORY/result.json');react_fit=read(OUT/'standalone_heads_native/REACT/result.json')
    if memory_fit['status']!='INSUFFICIENT_INFORMATIVE_DATA' or any(memory_fit['informative'].values()):raise AssertionError('unexpected MEMORY eligibility')
    for path,digest in react_fit['eligibility']['source_sha256'].items():
        if sha(path)!=digest:raise AssertionError('canonical labels changed after fitting')
    stop3=deltas['23']['HOTA']<0 and deltas['23']['AssA']<0
    if not stop3:raise RuntimeError('clean REACT not worse on both metrics: execute conditional augmentation before finalization')
    memory={'status':'COMPLETE','videos':baselines,'M6':{'status':'NOT_TRAINED_ZERO_INFORMATIVE_EXAMPLES','source':memory_fit,'metrics':None},
        'all_controllers_MATCH':'same permanent B2 with native learned second-round commit logic','seed':'branch-local native20261006+video',
        'representation_scope':'raw GMT ReID vectors; adapted EMA update formulas, not full reproductions of normalized external tracking pipelines',
        'M2_prediction_identical_to_M0_on_both_complete_sequences':True,'learned_MEMORY_value_claim':False,
        'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Memory quality/representation matters even when sampled single-write marginal utility ties. Confidence gating improves both train24 and val23; latest10 exactly reproduces native M0. Fixed EMA is strongest here on24, new-weight0.8 momentum on23; neither pattern establishes one universal winner or rescues untrained M6.'}
    save(REPORTS/'MEMORY_BASELINE_COMPARISON.json',memory)
    data=read(REPORTS/'REACTIVATION_CANONICAL_DATA_AUDIT.json');data.update(status='COMPLETE',fit=react_fit,
        complete_closed_loop={v:{'metrics':r['metrics'],'actions':r['actions'],'prediction_sha256':r['prediction_sha256'],'delta_vs_native_B2':deltas[v]} for v,r in heads.items()},
        THIRD_STOP_CONDITION=True,learned_REACTIVATION_enabled=False,
        what_did_we_learn='Canonical coverage and relative-state supervision repair the old zero-example protocol, but the fixed head improves train24 while harming complete val23. Training accuracy/coverage do not establish rescue; disable learned REACTIVATION under the requested third stop.')
    save(REPORTS/'REACTIVATION_CANONICAL_DATA_AUDIT.json',data)
    save(REPORTS/'CLOSED_LOOP_STATE_AUGMENTATION.json',{'status':'NOT_RUN_REQUESTED_STOP_CONDITIONS','clean_REACT_fit_completed':True,
        'clean_complete_closed_loop_deltas':deltas,'user_third_stop_triggered':True,
        'unified_eligible':False,'shared_corruption_training_completed':False,'clean_plus_state_corruption_comparison_completed':False,
        'real_state_perturbations_generated':0,'feature_noise_with_old_labels_used':False,
        'reason':'Original structured MATCH prerequisite fails, MEMORY has no informative training examples, and canonical clean REACT loses both val23 primary metrics. Obey the user third stop; do not expand a failing learned component.',
        'planned_if_eligible':['actual wrong previous MATCH','missed association/false birth','missing write','contaminated bank anchor','missing stale candidate'],
        'required_future_method':'perturb actual mutable history/memory/pool, rebuild online features, regenerate live paired targets; never feature noise retaining old labels',
        'late_stop_document_is_preregistration':False,'source_stop_rule':'original user FINAL GOAL THIRD STOP CONDITION, before all runs',
        'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Self-induced state generalization remains unresolved. Canonical clean training is tested and fails validation; no robustness benefit is claimed from an augmentation experiment that was not run.'})
    architecture=read(REPORTS/'TYPED_JEV_ARCHITECTURE.json');architecture.update(status='PROTOTYPE_INPUT_CONTRACT_CHECKED_STANDALONE_REACT_ONLY_TRAINED',
        standalone_typed_parameters=39712,standalone_total_runtime_parameters=73792,
        standalone_core_sharing=False,standalone_MATCH='separate permanently frozen34080-param B2',
        joint_shared_training_completed=False,unified_training_eligible=False,
        what_did_we_learn='Three typed adapters fit in39,712 parameters. The bounded standalone REACT diagnostic uses a separate frozen B2 plus a copied typed core (73,792 runtime parameters), so it is not evidence for a trained shared lifecycle reasoner.')
    save(REPORTS/'TYPED_JEV_ARCHITECTURE.json',architecture)
    gating=read(REPORTS/'FANCY_GATING_AUDIT.json');gating['native_followup']='NATIVE_HELDOUT_MECHANISM_AUDIT.json'
    gating['interpretation_amendment']='Original development gate FAIL is unchanged. Corrected native fixed-weight controls show a material global-reassignment contribution on heldout02; it is too strong to attribute every native gain to gating. This does not establish universal three-action necessity or complete Lifecycle success.'
    save(REPORTS/'FANCY_GATING_AUDIT.json',gating)
    ablation={'status':'UNIFIED_INELIGIBLE_BOUNDED_COMPONENT_DIAGNOSTICS_COMPLETE',
        'conditions_completed':['GMT','score-only scalar threshold','full64D dynamic threshold','binary JEV','frozen B2 three-action MATCH','frozen B2 forced-new/accept removal','native validation-only mechanism control','MATCH+fixed MEMORY M0-M5','MATCH+standalone canonical REACT'],
        'new_generic_MLP':'NOT_RETRAINED_FOR_INELIGIBLE_UNIFIED_STAGE; inherited Phase V C2 collapse is a separate historical diagnostic, not a new fair-budget result',
        'MATCH_plus_learned_MEMORY':'NOT_TRAINED_ZERO_INFORMATIVE_EXAMPLES','Full_Lifecycle_JEV':'NOT_TRAINED_PREREQUISITES_FAIL','shared_clean_plus_aug':'NOT_RUN_USER_THIRD_STOP',
        'standalone_REACT':{'fit':react_fit,'closed_loop_deltas':deltas,'enabled':False},
        'structured_MATCH_original_gate':'FAIL','learned_MEMORY_informative':False,'canonical_REACT_informative':True,
        'full_exceeds_MATCH_claim':'NOT_ESTABLISHED','fair_joint_data_seed_optimizer_parameter_ablation_completed':False,
        'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Component evidence does not support a Full Lifecycle model: global rematching has context-dependent value, fixed memory heuristics help, learned MEMORY has no targets, and learned REACT fails validation. Do not fill missing Full/MLP rows with inherited or fabricated metrics.'}
    save(REPORTS/'UNIFIED_LIFECYCLE_ABLATION.json',ablation)
    held=read(REPORTS/'HELDOUT_RESULTS.json');native=held['conditions']['G5_NATIVE_TRANSITION'];contract=read(REPORTS/'NATIVE_TRANSITION_CONTRACT_AUDIT.json')
    phase5=Path('/data1/liuyeqiang/WWW_jev_phase5')
    phase5_head=subprocess.check_output(['git','-C',str(phase5),'rev-parse','HEAD'],text=True).strip()
    phase5_dirty=subprocess.check_output(['git','-C',str(phase5),'status','--porcelain'],text=True).strip()
    if phase5_head!='a37083dac0a23eb3846cb252eeb975982aaaabc9' or phase5_dirty:raise AssertionError('Phase V evidence changed')
    report={'status':'COMPLETE','PHASE6_GO':False,'decision':'NO-GO','FULL24_AUTHORIZED':False,'OFFICIAL_TEST_AUTHORIZED':False,
        'B2_permanent_checkpoint_sha256':sha(B2),'B2_original_research_anchor_preserved':True,'PhaseV_base_preserved_clean':True,
        'gate_results':{'structured_MATCH_original_development_gate':'FAIL','Full_at_least_MATCH_or_both_learned_extensions_positive':'FAIL',
            'multiple_native_heldouts_positive':native['positive_sequences']>=2,'preregistered_all_three_positive':native['preregistered_gate_pass'],
            'full_native_lifecycle_integration':'NOT_ESTABLISHED','native_MATCH_commit_and_MEMORY_eligibility_on11events':contract['opt_in_corrected_native_MATCH_contract'],
            'no_official_TEST_used':True},
        'heldout_native_positive_sequences':native['positive_sequences'],'heldout_native_pooled_metrics':native['pooled_metrics'],
        'heldout_native_pooled_delta_vs_GMT':native['pooled_delta_vs_GMT'],'stop_conditions':{'first_exact_equivalence':False,'second_learned_MEMORY_removed':True,'third_learned_REACT_disabled':True},
        'questions':{'MATCH_structure_or_gating':'Native global reassignment has causal value on02, beyond second gating; pooled mechanism advantage positive. Original development three-action>binary gate fails and gains are context-dependent, so universal necessity is not established.',
            'can_MEMORY_be_rescued':'Fixed confidence/EMA/momentum rules help complete sequences, but this20-example READ-driven protocol yields no informative learned M6 examples. Memory itself is not disproved; learned MEMORY is not rescued.',
            'can_REACTIVATION_be_rescued':'Corrected canonical data give16 train/9 val informative examples, yet learned head loses val23 HOTA/AssA; disable it. This bounded failure does not disprove the concept.',
            'does_unified_beat_MATCH_only':'NOT_ESTABLISHED: unified training correctly withheld because prerequisites fail.',
            'WWW_full_lifecycle_story_supported':False,'should_Full24_be_authorized':False},
        'limitations':['single fixed seed and bounded enriched diagnostic samples','video01 is reused development only','heldouts are TRAIN-controller heldouts; GMT backbone may have seen them',
            'raw TRAIN GT duplicate override retained consistently from prior evaluation; these are diagnostic scores, not official TEST',
            'native MATCH logic is verified separately from unimplemented production relative-REACT integration',
            'no clean+state-corruption comparison or fair unified ablation claimed when skipped'],
        'what_did_we_learn':'Preserve the frozen positive anchor and the negative evidence. Native transition fidelity changes scientific conclusions; real global reassignment and fixed memory quality rules have value, but learned lifecycle components and all original gates do not support the proposed full WWW story.'}
    save(REPORTS/'FINAL_PHASE6_GO_NO_GO.json',report);protect_anchor()
    print(json.dumps({'decision':'NO-GO','REACT_validation_delta':deltas['23'],'native_heldout_positive':native['positive_sequences'],'Full24':False}))

if __name__=='__main__':main()
