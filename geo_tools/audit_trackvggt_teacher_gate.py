"""Close the teacher screening gate only after all fixed hypotheses finish.

Positive numerical directions never automatically admit a teacher: replication
and actual identity evidence would still be required. This does not close the
overall goal or substitute ranking changes for actual joint solver repairs.
"""
from common import *


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');required={};pending=[];runs={};reference=None
    plans=['trackvggt_phase2_ag_fit_v1','trackvggt_v3_ag_fit_v1','trackvggt_qd_fit_v1',
        'trackvggt_crop_fit_v1','trackvggt_qd_native_control_v1','trackvggt_cycle_fit_v1']
    for name in plans:
        plan_path=ROOT/'manifests'/(name+'.json');plan=json.loads(plan_path.read_text())
        for stages in plan['queues'].values():
            for stage in stages:required[stage['id']]=stage
    for tag in required:
        path=runtime/'models'/tag/'manifest.json'
        if not path.exists():pending.append(tag);continue
        value=json.loads(path.read_text())
        if value['status'] not in ('COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION','COMPLETE_MATCHED_REAL_CYCLE_CONTINUATION'):
            raise ValueError('invalid teacher completion status: '+tag)
        if value['provenance']['command']!=required[tag]['command']:
            raise ValueError('actual fitted command differs from the registered stage: '+tag)
        for name,digest in value['artifacts'].items():
            if sha256(path.parent/name)!=digest:raise ValueError('fitted teacher artifact changed: '+tag+'/'+name)
        if value['validation_metrics']['requests']!=300 or value['validation_metrics']['matched_eligible']!=100:
            raise ValueError('teacher screening denominators changed')
        validation=json.loads((path.parent/'validation.json').read_text())['rows']
        signature=[[r[k] for k in ('key','query_gt_known','top8_positive','known_unmatched','original_native_top1_state','actual_frozen_gmt_primary_state')] for r in validation]
        if reference is None:reference=signature
        if len(signature)!=300 or signature!=reference:raise ValueError('registered teacher validation membership/order/labels differ')
        runs[tag]=dict(path=str(path),sha256=sha256(path),value=value)
    if pending:
        proof=dict(status='PENDING_REGISTERED_INDEPENDENT_TEACHER_HYPOTHESES',pending=pending,
            completed=len(runs),required=len(required),teacher_admitted=False,overall_goal_complete=False,provenance=provenance())
        save_json(ROOT/'results/trackvggt_teacher_gate_v1.json',proof)
        print(json.dumps({k:v for k,v in proof.items() if k!='provenance'}));return
    if len(runs)!=56:raise ValueError('fixed teacher run inventory differs from expected52 no-cycle plus4 continuations')
    def metric(tag,field='Recall1',shuffled=False):
        return runs[tag]['value']['query_shuffle_metrics' if shuffled else 'validation_metrics'][field]
    ag=lambda b,mode:f'ag_{b}_{mode}_mean_k24_seed20261015_v3_1000_v1'
    qd=lambda b,mode:f'qd_{b}_{mode}_k24_seed20261015_v3_1000_v1'
    cycle=lambda b,weight:f'cycle_{b}_lambda{weight}_seed20261015_v3_plus200_v1'
    comparisons=[dict(question='V3 identity AG versus matched ROI',proposed=ag('vggt','C'),control=ag('vggt','roi')),
        dict(question='Track versus global See-Query',proposed=qd('vggt','track'),control=qd('vggt','global')),
        dict(question='Track QD versus matched DINO track',proposed=qd('vggt','track'),control=qd('dinov2','track')),
        dict(question='Track QD versus QD-capacity nativeMLP',proposed=qd('vggt','track'),control='native_mlp_qd_capacity_seed20261015_v3_1000_v1'),
        dict(question='Equal-update temporal consistency',proposed=cycle('vggt','0p1'),control=cycle('vggt','0p0')),
        dict(question='Temporal VGGT versus matched temporal DINO',proposed=cycle('vggt','0p1'),control=cycle('dinov2','0p1'))]
    for row in comparisons:
        row.update(proposed_Recall1=metric(row['proposed']),control_Recall1=metric(row['control']))
        row['strictly_higher_primary_seed_Recall1']=row['proposed_Recall1']>row['control_Recall1']
        row['scope']='Numerical direction only, single trained seed,100 matched-positive ranks; not independent significance, replication or actual final-ID repair.'
    for mode in ['full','local','combined']:
        vggt=f'ag_vggt_C_crop224_{mode}_k24_seed20261015_1000_v1'
        dino=f'ag_dinov2_C_crop224_{mode}_k24_seed20261015_1000_v1'
        comparisons.append(dict(question='Matched224 crop '+mode,proposed=vggt,control=dino,
            proposed_Recall1=metric(vggt),control_Recall1=metric(dino),strictly_higher_primary_seed_Recall1=metric(vggt)>metric(dino),
            scope='Same256 fit and original300 validation. Separate from full3000-fit QD/AG.' ))
    paired={}
    for backbone in ['vggt','dinov2']:
        path=ROOT/'results'/f'trackvggt_cycle_{backbone}_paired_v1.json';value=json.loads(path.read_text())
        if value['status']!='PASS_IDENTICAL_SOURCE_AND_MATCHED_TEMPORAL_CONTROLS':raise ValueError('cycle pairing is not independently audited')
        if value['source_checkpoint_sha256']!=runs[cycle(backbone,'0p1')]['value']['source_checkpoint_sha256']:
            raise ValueError('cycle pairing references a different initial teacher')
        paired[backbone]=dict(path=str(path),sha256=sha256(path),known_cycle_ranking_fixes=value['known_ranking_fix_keys'],
            known_cycle_ranking_harms=value['known_ranking_harm_keys'],uncertified_to_correct=value['uncertified_to_correct_keys'])
    shuffle={mode:dict(original_Recall1=metric(qd('vggt',mode)),history_shuffle_Recall1=metric(qd('vggt',mode),shuffled=True),
        history_shuffle_reduces_Recall1=metric(qd('vggt',mode),shuffled=True)<metric(qd('vggt',mode))) for mode in ['global','track']}
    # These are explicit user requirements, not a new fitted validation threshold.
    necessary=all(r['strictly_higher_primary_seed_Recall1'] for r in comparisons[:6]) and shuffle['track']['history_shuffle_reduces_Recall1']
    status='NEEDS_REPLICATION_AND_MANUAL_GATE_REVIEW' if necessary else 'NO_UNIQUE_VGGT_TEACHER_EVIDENCE'
    proof=dict(status=status,completed_registered_runs=len(runs),pending=[],comparisons=comparisons,
        paired_temporal_evidence=paired,query_shuffle=shuffle,native_Recall1=.66,
        original_validation_queries=300,matched_positive_ranking_queries=100,trained_seeds_actually_executed=[20261015],
        teacher_admitted=False,student_disposition='CONDITIONAL_NOT_EXECUTED_RELATION_GATE_FAILED' if not necessary else 'NOT_ADMITTED_PENDING_REPLICATION_AND_VALID_IDENTITY_EVIDENCE',
        cycle_lambda0p2_disposition='NOT_EXECUTED_NO_VALIDATED_INCREMENTAL_IDENTITY_BENEFIT; no arbitrary increase',
        evidence_limits=['Single held-out association scene, not multi-scene replication.',
            'Original300 includes0 certified actual primary solver errors; row-ranking fixes do not prove Hungarian/Bank/final-ID repairs.',
            'Some VGGT ranking fixes differ from a particular matched DINO mode; overlaps and harms must be read per-control, not asserted universally.',
            'Raw cycle generic seed14 is preserved; actual training seed15 is resolved in paired audits. Native inference seed14 is distinct.',
            'Historical periodic training Recall/margin curves were not recorded; only actual retained150/1000 fit endpoints and logged loss/gradient curves are available.'],
        runs={tag:{k:v for k,v in item.items() if k!='value'} for tag,item in runs.items()},
        scientific_success=False,student_success=False,paper_ready=False,overall_goal_complete=False,
        remaining_overall_work='Complete fresh full native/Prior system/retrieval/strict5 audit and final requirement/limitation reports. A negative teacher gate does not complete the overall goal.',
        provenance=provenance())
    save_json(ROOT/'results/trackvggt_teacher_gate_v1.json',proof)
    print(json.dumps(dict(status=status,completed=len(runs),comparisons=comparisons,teacher_admitted=False)))


if __name__=='__main__':main()
