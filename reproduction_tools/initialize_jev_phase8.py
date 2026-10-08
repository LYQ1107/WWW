"""Freeze the Phase VIII split and gates before reading new outcome metrics."""
import datetime,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/JEV_PHASE8'
OUT=Path('/home/liuyeqiang/WWW_jev_phase8_runtime/20261008_v1')
CACHE=Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text()==text,('immutable preregistration already exists',p)
    else:p.write_text(text)

def main():
    scenes={};counts={}
    for line in(CACHE/'index.jsonl').open():
        d=json.loads(line);v=int(d['video_id']);counts[v]=counts.get(v,0)+1
        part=Path(d['metadata']['file_name']).parts[-3].split('_View')[0]
        scenes.setdefault(v,re.sub(r'^\d+','',part))
    frozen={'status':'PREREGISTERED_BEFORE_NEW_CORRECTIVE_OUTCOMES','base_phase7':'ecc0e63f544cbbc272797c17ea285b0b8cce5f80',
      'train_videos':[12,13,14,16],'validation_videos':[17,18,19],'controller_heldout_videos':[20,21,22],
      'explicit_historically_used_videos':[1,2,3,4,5,6,7,8,9,10,11,15,23,24],
      'historical_additions':{'8':'native early closed-loop pilot; retained code/docs','15':'prior Full H8 worker, documented in JEV_RNG_V4_CURRENT_HEAD_RUN_20261006.md'},
      'scene_grouping_rule':'same scene token in official filenames stays in one Phase VIII partition; tokens describe scene families, not proven independent physical recordings',
      'scene_by_video':scenes,'exogenous_cached_payloads_by_video':counts,
      'history_limitations':'all TRAIN has cached GMT perception and foundation training exposure; video22 shares basketball family with previously used video23; these are preregistered new MATCH-controller holdouts, not unseen foundation/test data; no prior positive metrics used to choose partitions',
      'seeds':[20261008,20261009,20261010],'primary_seed':20261008,'scan_factual_policy':'GMT OFF, live native mutable state and frozen perception/scoring/commit functions',
      'annotation_contract':'official TRAIN only; cache frame/view zero-based, annotations one-based; IoU>=0.5 per-image one-to-one boxes at the actual original image size',
      'historical_identity_anchor':{'minimum_preceding_known_observations':2,'minimum_purity':1.0,'first_known_GT_must_equal_modal_GT':True,'unanchored_or_mixed':'UNKNOWN, not a negative identity label','multiple_predicted_IDs_for_one_GT':'retain all aliases and report ambiguity; do not compare identity integers'},
      'event_grouping':'same video and target GT; correction events connected within 32 frames and overlapping candidate/conflict identities are one group; group weights sum to one; report video/scene units and Kish ESS separately',
      'bounded_snapshots':{'max_corrective_groups_per_video':32,'max_hard_negative_events_per_video':8,'selection':'chronological first event per eligible correction group; hard negatives first known-correct proposal with top1-top2 gap<=0.1 in each of four exogenous frame bins, maximum two per bin'},
      'native_horizons':[8,16,32],'primary_horizon':32,'future_policy':'common live GMT OFF recomputed after each actual mutated commit; no frozen teacher future action map',
      'primary_utility':'correct anchored observations - wrong anchored observations - 5*false_merge_tracks - 0.25*false_births; report target and all-row externalities, unknown/tie/censor separately',
      'utility_sensitivity':{'false_birth_weights':[0.0,0.25,1.0],'false_merge_weights':[2.0,5.0],'wrong_identity_weights':[1.0,2.0],'model_selection':'primary weights fixed; other weights are diagnostics and cannot select a favourable heldout run'},
      'formal_data_gate':{'verified_train_corrective_events':50,'verified_validation_corrective_events':20,'train_videos_with_verified_events':2,'validation_videos_with_verified_events':2,'independent_train_event_groups':5,'independent_validation_event_groups':3,'corrective_actions':'correct history candidate exists, global legal, actual native commit improves immediate identity and H32 utility against CONTROL; benefit survives removing the birth penalty','native_control_commit_full_state_parity':'PASS','GT_future_input_leakage':False,'candidate_choice_outcomes_minimum':2},
      'failure_gate':'BLOCKED_DATA_OPPORTUNITY or BLOCKED_NATIVE_CANDIDATE_INTERFACE; no model/large dataset/heldout search may bypass failed formal gates',
      'learning_curves_epochs':[5,20,50,100],'data_scaling_group_fractions':[0.25,0.5,0.75,1.0],
      'capacity_targets':[8000,34000,128000,500000],'optimizer':{'type':'AdamW','lr':0.001,'weight_decay':0.0001,'batch_size':32,'gradient_clip':5.0,'fixed_last_epoch':100,'lr_schedule':'cosine; same scheduled update counts in like-for-like comparisons'},
      'normalization':'train-only feature statistics; same exact transform online/offline; raw, standardized and LayerNorm diagnostic controls',
      'sampling_losses':['natural_CE','stratified_hard_CE','stratified_hard_focal_gamma2','candidate_pairwise_ranking','utility_H32_regression','correctness_plus_utility'],
      'fairness':'all candidate models same legal candidate evidence/assignment/native commit, data, calibration, optimiser/update budgets; parameter gap<=1% near B2 target where feasible, measured MAC and latency reported independently',
      'architecture_gate':{'pooled_HOTA_gain_over_strongest_preregistered_ordinary_baseline':0.1,'pooled_AssA_gain':0.2,'positive_controller_heldout_videos_minimum':2,'correction_not_cancelled_by_regression':True,'extra_policy_plus_assignment_latency_p95_ms':10.0,'extra_GPU_memory_MiB':128,'statistical_units':'whole videos/scene families, never overlapping windows; three seeds when data supports fitting'},
      'heldout_release_rule':'do not run candidate model or tune thresholds on video20/21/22 until train/validation gates and model selections are sealed',
      'baseline_tags':['GMT_OFF','FIXED_RULE','DYNAMIC_THRESHOLD','CANDIDATE_MLP','CANDIDATE_DEEPSETS','STATE_JEV','FROZEN_B2','CANDIDATE_JEV','CANDIDATE_JEV_CAUSAL_UTILITY'],
      'Full24_allowed':False,'official_TEST_read':False,'million_record_build_allowed':False,'Unified_allowed_without_three_independent_gates':False,
      'cache_index_sha256':sha(CACHE/'index.jsonl'),'foundation_sha256':json.loads((REPORTS/'STORAGE_BEFORE.json').read_text())['foundation_sha256'],
      'protected_B2_sha256':'f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7',
      'goal_sha256':sha(ROOT/'docs/PHASE8_FINAL_GOAL.md')}
    train={scenes[v]for v in frozen['train_videos']};val={scenes[v]for v in frozen['validation_videos']};held={scenes[v]for v in frozen['controller_heldout_videos']}
    assert not(train&val or train&held or val&held)
    write(REPORTS/'PREREGISTRATION.json',json.dumps(frozen,indent=2,sort_keys=True)+'\n')
    write(ROOT/'docs/PHASE8_RESEARCH_PLAN.md',"""# Phase VIII frozen research plan

P-1 completed first: 656 checkpoint/state files audited, 37.8 GiB retained, zero safe orphan deletion, eight cleanup tests PASS. /home runtime is separate from space-constrained /data1. Prior Phase V/VI/VII evidence and all negative experiments are protected.

The immutable machine protocol is `reports/JEV_PHASE8/PREREGISTRATION.json`. Train: **12/13/14/16** (path/football); validation: **17/18/19** (wood/park); controller-heldout: **20/21/22** (road/bridge/basketball). Same scene families never cross this new split. Historical inspection additionally excludes video08 (pilot) and video15 (prior Full H8 worker). All official TRAIN has foundation/perception exposure; basketball22 also shares a family with earlier video23. No claim of unseen foundation or official TEST generalization is made. New outcome metrics have not been inspected to choose this split.

Phase A runs only necessary GMT-OFF native proposal/candidate audits on the fixed train/validation videos. Prefix historical anchors require two preceding known matches and complete purity, and retain duplicate predicted-ID aliases for one GT. Distinguish correct proposal, missing candidate, infeasible conflict, legal alternative, successful immediate commit and H8/H16/H32 benefit. Only sparse difficult-event indexes and bounded snapshots are retained; no unconditional lifelong detection-by-history matrix archive. Current proposals/scores/masks and RNG are exact, and future states mutate under common live GMT OFF.

For formal fitting, adopt the suggested 50 train / 20 validation verified corrective events, at least two contributing videos per partition, and at least five / three temporally separate conflict groups. A group carries total weight one. Correction must be legal, native-committed, immediately beneficial and beneficial at H32 even with birth penalty zero. Full-field CONTROL/factual state and committed IDs must match. Missing GT, unstable anchors, aliases, ties and censoring stay explicit. Sparse availability counts alone cannot pass this gate.

If the gate passes, construct Native Corrective v2 with natural distribution, hard corrections and predeclared hard negatives. Separate candidate correctness from long-term utility. Then run C1 tiny overfit; C2 25/50/75/100% independent-group data curves at fixed update budget; C3 5/20/50/100 epoch checkpoints/curves; C4 actual ~8K/34K/128K/500K capacities; C5 predeclared sampling/loss choices; C6 train-stat normalization controls. All comparisons use identical effective data and targets when diagnosing architecture/capacity. Only best/last and genuinely necessary milestone weights stay in the runtime.

Only after supervision and native interface gates pass, develop Candidate MLP/DeepSets and Candidate JEV with identical evidence, legal actions and shared assignment/commit operator. IDs are metadata, not features. Measure permutation/mask/NEW/duplicate/collision/commit/state/RNG/no-GT contracts. Freeze model selection/calibration on validation, then release the three heldout videos for live closed loops and proper pooled TrackEval. Include seeds 20261008/09/10 if support permits, video-level uncertainty and a bounded identity-index query audit with oracle alignment limitations stated.

A MATCH architecture GO requires >=+0.1 pooled HOTA and >=+0.2 AssA over the strongest preregistered ordinary baseline, >=2 positive heldout videos, beneficial correction not erased by regression, p95 added policy/assignment <=10 ms and extra GPU memory <=128 MiB. Statistical uncertainty is a separate requirement and cannot be fabricated from overlapping events. Candidate conditioning alone is not JEV-specific evidence. Failed gates produce BLOCKED/NO-GO with conditional phases NOT_RUN.

MEMORY/relative REACT/Unified remain conditional: no unidentifiable WRITE labels or unverified relative resolver is promoted into training. A MATCH GO does not authorize Full24 or official TEST. Every completed stage reports WHAT DID WE LEARN and is compactly published; raw fork/score/prediction evidence remains local.
""")
    write(ROOT/'docs/PHASE8_FAIR_BASELINE_PROTOCOL.md',"""# Phase VIII fair comparison protocol

The nine frozen baseline tags in PREREGISTRATION cover GMT OFF, fixed rules, a strong normalized learnable threshold, candidate MLP, candidate DeepSets, existing state JEV, protected B2, Candidate JEV and its causal-utility variant. Frozen historical B2 is an anchor with different historical labels; freshly trained models share the exact v2 split, effective groups, online fields and correctness targets. Neither a frozen historical anchor nor state-only input is falsely called an identical-data candidate architecture experiment.

Candidate MLP/DeepSets/JEV receive the same 64 state and 12 real candidate fields, raw GMT scores, semantic NEW, legal masks, shared native assignment operator and identical downstream MEMORY/stale/birth transitions. Preserve all candidates, suppress actual duplicate IDs consistently and enforce one identity per row/view while allowing lawful multi-camera reuse. The new direct interface must be tested against actual production operators; copying research resolver outputs is insufficient. Any score/cost change applies equally to ordinary and JEV candidate controllers. Original GMT OFF and frozen B2 behavior remain separately reported.

Use train-only statistics, validation-only temperature calibration and no heldout threshold/sequence selection. Main comparisons aim for <=1% parameter difference around the protected B2 count, with independent compute/latency measurements. Data curves sample nested independently grouped events and keep total updates fixed; epoch curves hold data/model fixed; capacity curves hold data/target/budget fixed; correctness versus utility changes targets on identical networks. Sampling uses natural/stratified-hard training variants but validation always includes natural and independently reported frozen difficult strata. Pairwise ranking, focal gamma2, utility and multitask choices are frozen before validation outcomes.

H8/H16/H32 forks start from identical complete mutable state and RNG, never GT-conditioned runtime actions. Final full-video runs recompute each controller's future actions and truly commit state. Metrics include HOTA/AssA/IDF1/IDSW/MOTA, explicitly sourced Frag, candidate MRR/Top1, correction/regression, false merges/births, temporal wrong-identity episodes, contamination, trigger/NEW frequencies, p50/p95 latency and memory. TrackEval pooling uses its true sequence-combination operator. Seeds are 20261008/09/10, statistical repetitions are independent videos/scene groups. If a gate fails, record NOT_RUN instead of fabricating fair-model or heldout measurements.
""")
    print(json.dumps({'status':frozen['status'],'preregistration_sha256':sha(REPORTS/'PREREGISTRATION.json'),'train':frozen['train_videos'],'validation':frozen['validation_videos'],'heldout':frozen['controller_heldout_videos']}))

if __name__=='__main__':main()
