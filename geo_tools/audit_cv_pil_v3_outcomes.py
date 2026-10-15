"""Separate real original-error repairs from ranking/avoidance-only changes.

Offline report over an immutable evaluation CSV; no model, video or GT matcher.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from cv_pil_dev12_assets import require
from cv_pil_native_handoff import sha256

ARMS=('C0','C1','C2','C3','C4')


def read_rows(path):
    output=[]
    for raw in csv.DictReader(Path(path).open()):
        r=dict(raw);r['key']=json.loads(r['key']);r['person']=int(r['person']) if r['person'] else None
        for name in ['positive_covered',*[arm+'_primary' for arm in ARMS],*[arm+'_rank' for arm in ARMS]]:
            require(r[name] in ('True','False',''), 'Unrecognized certification value')
            r[name]={'True':True,'False':False,'':None}[r[name]]
        for arm in ARMS:r[arm+'_accepted_id']=int(r[arm+'_accepted_id']) if r[arm+'_accepted_id'] else None
        output.append(r)
    return output


def targeted_audit(rows):
    require(len({tuple(r['key']) for r in rows})==len(rows), 'Repeated physical query')
    errors=[r for r in rows if r['source_primary_state'] in ('wrong_link','missed_known_positive')]
    correct=[r for r in rows if r['source_primary_state'] in ('correct_link','correct_reject')]
    require(all(r['C0_primary'] is False for r in errors) and
            all(r['C0_primary'] is True for r in correct), 'Source primary certification conflicts with C0 replay')
    reachable=[r for r in errors if r['positive_covered']]
    methods={};fixes={};harms={};unknown={}
    for arm in ARMS:
        fixed=[r for r in reachable if r[arm+'_primary'] is True and r[arm+'_accepted_id'] is not None]
        damaged=[r for r in correct if r[arm+'_primary'] is False]
        unresolved=[r for r in correct if r[arm+'_primary'] is None]
        fixes[arm]={tuple(r['key']) for r in fixed};harms[arm]={tuple(r['key']) for r in damaged};unknown[arm]={tuple(r['key']) for r in unresolved}
        methods[arm]=dict(certified_reachable_original_primary_errors_repaired=len(fixed),
            repaired_video_person_groups=len({(12,r['person']) for r in fixed}),
            originally_correct_primary_known_harms=len(damaged),originally_correct_to_unknown=len(unresolved),
            net_reachable_repairs_minus_known_harms=len(fixed)-len(damaged),
            fixed_query_keys=[r['key'] for r in fixed], harmed_query_keys=[r['key'] for r in damaged],
            rejected_without_legal_positive=sum(not r['positive_covered'] and r[arm+'_accepted_id'] is None for r in errors),
            final_closed_loop_identity_repairs=None)
    keyed={tuple(r['key']):r for r in rows};comparisons={}
    for baseline in ('C0','C1','C2','C4'):
        extra=fixes['C3']-fixes[baseline];lost=fixes[baseline]-fixes['C3'];by_person=defaultdict(lambda:dict(extra_original_repairs=0,lost_original_repairs=0,additional_known_harms=0,avoided_known_harms=0))
        for k in extra:by_person[str(keyed[k]['person'])]['extra_original_repairs']+=1
        for k in lost:by_person[str(keyed[k]['person'])]['lost_original_repairs']+=1
        for k in harms['C3']-harms[baseline]:by_person[str(keyed[k]['person'])]['additional_known_harms']+=1
        for k in harms[baseline]-harms['C3']:by_person[str(keyed[k]['person'])]['avoided_known_harms']+=1
        comparisons['C3_vs_'+baseline]=dict(extra_original_error_repairs=len(extra),lost_original_error_repairs=len(lost),
            net_original_error_repairs=len(extra)-len(lost),
            extra_repaired_video_person_groups=len({keyed[k]['person'] for k in extra}),
            net_positive_original_repair_person_groups=sum(
                g['extra_original_repairs']>g['lost_original_repairs'] and
                g['extra_original_repairs']-g['lost_original_repairs']-g['additional_known_harms']+g['avoided_known_harms']>0
                for g in by_person.values()),
            additional_known_harm_queries=len(harms['C3']-harms[baseline]),
            additional_correct_to_unknown=len(unknown['C3']-unknown[baseline]),
            avoided_ordinary_model_harms=len(harms[baseline]-harms['C3']),
            by_person=dict(by_person),extra_fix_keys=[list(k) for k in sorted(extra)],lost_fix_keys=[list(k) for k in sorted(lost)])
    c32=comparisons['C3_vs_C2'];c34=comparisons['C3_vs_C4']
    conditions=dict(reachable_original_errors_present=bool(reachable),
        C3_more_original_error_repairs_than_C2=c32['net_original_error_repairs']>0,
        C3_net_original_repairs_in_at_least2people=c32['net_positive_original_repair_person_groups']>=2,
        C4_does_not_explain_original_repairs=c34['net_original_error_repairs']>0,
        no_increase_in_known_harm_count_vs_C2=len(harms['C3'])<=len(harms['C2']),
        no_increase_in_correct_to_unknown_count_vs_C2=len(unknown['C3'])<=len(unknown['C2']),
        C3_positive_net_repairs_over_original_GMT=methods['C3']['net_reachable_repairs_minus_known_harms']>0,
        C3_positive_net_repairs_over_original_in2people=comparisons['C3_vs_C0']['net_positive_original_repair_person_groups']>=2)
    if not reachable:status='INSUFFICIENT_NATIVE_OPPORTUNITY'
    elif c32['net_original_error_repairs']<=0:status='NO_UNIQUE_PREDICTIVE_EFFECT'
    elif c34['net_original_error_repairs']<=0:status='CAMERA_CONDITION_VALUE_NOT_ESTABLISHED'
    else:status='OFFLINE_DEVELOPMENT_POSITIVE'
    return dict(status=status,original_errors=len(errors),legal_positive_reachable_errors=len(reachable),
                methods=methods,comparisons=comparisons,algorithm_gate_conditions=conditions,
                algorithm_gate_pass=all(conditions.values()),native_online_run_admitted=False,
                native_interface_and_OFF_attestation='REQUIRED_AFTER_ALGORITHM_GATE_BEFORE_ONLINE',
                qualification='Single-step full primary Hungarian under original frozen state. Rank gains, avoiding C2-induced harms and correct rejection without a legal positive are separately counted; none proves final closed-loop identity improvement.',
                scientific_success=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'Cannot overwrite prior mechanism audit')
    report=targeted_audit(read_rows(a.csv));report['query_csv_sha256']=sha256(a.csv)
    report['source_sha256']=sha256(Path(__file__))
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(report['status'])


if __name__=='__main__':main()
