"""Publish compact availability evidence and conflict-group independence."""
import gzip,json
from collections import Counter
from jev_phase8_common import *

def conflict_groups(events):
    parent=list(range(len(events)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    ordered=sorted(range(len(events)),key=lambda i:(events[i]['key'][1],events[i]['key'][2],events[i]['row']))
    for pos,i in enumerate(ordered):
        a=events[i];ai=set(a['correct_candidate_ids'])|{a['proposal_track_id']}
        for j in reversed(ordered[:pos]):
            b=events[j]
            if a['key'][1]-b['key'][1]>32:break
            bi=set(b['correct_candidate_ids'])|{b['proposal_track_id']}
            if a['offline_GT']==b['offline_GT']or ai&bi:parent[find(i)]=find(j)
    roots={r:n for n,r in enumerate(sorted({find(i)for i in range(len(events))}))}
    return {tuple(e['key'])+(e['row'],):f'video{e["key"][0]:02d}_conflict{roots[find(i)]:03d}'for i,e in enumerate(events)}

def main():
    protect();report={'status':'COMPLETE_ORACLE1_ORACLE2','binding':binding(),'videos':{},'raw_manifest':[],'formal_data_gate':'PENDING_ACTUAL_NATIVE_H32',
        'native_direct_candidate_deployment_parity':'NOT_VERIFIED','split_unchanged':True,'raw_evidence_uploaded':False}
    for v in(7,12,13,14,16,17,18,19):
        directory=OUT/('diagnostic_opportunities_v1'if v==7 else'opportunity_scan_v1')/f'video{v:02d}'
        x=json.loads((directory/'SCAN_RESULT.json').read_text())
        with gzip.open(directory/'hard_event_index.jsonl.gz','rt')as f:events=[json.loads(l)for l in f]
        groups=conflict_groups(events);save(directory/'CONFLICT_GROUP_INDEX.json',[{'key':list(k[:3]),'row':k[3],'group':g}for k,g in sorted(groups.items())])
        selected=[tuple(s['key'])+(r,)for s in x['bounded_snapshots']for r in s['selected_rows']]
        gc=Counter(groups[k]for k in selected);weights=[1/gc[groups[k]]for k in selected]
        old=OUT/('diagnostic_opportunities'if v==7 else'opportunity_scan')/f'video{v:02d}'/'SCAN_RESULT.json'
        oldx=json.loads(old.read_text())if old.exists()else None
        report['videos'][str(v)]={'role':x['role'],'counts':x['counts'],'snapshots':len(x['bounded_snapshots']),'selected_events':len(selected),
            'provisional_target_episode_groups':len(x['independent_error_groups']),'conflict_connected_groups':len(set(groups.values())),
            'selected_conflict_groups':len(gc),'selected_group_sizes':dict(gc),'selected_weight_Kish_ESS':sum(weights)**2/sum(w*w for w in weights)if weights else 0,
            'actual_production_OFF_checked_payloads':len(x['actual_production_OFF_checks']),'actual_production_OFF_max_feature_error':max(c['max_feature_error']for c in x['actual_production_OFF_checks']),
            'v0_v1_factual_prediction_exact_sha_parity':oldx['factual_predictions_sha256']==x['factual_predictions_sha256']if oldx else 'NO_V0_RUN',
            'factual_prediction_sha256':x['factual_predictions_sha256'],'scan_result_sha256':sha(directory/'SCAN_RESULT.json'),
            'snapshot_cap_scope':'max32 snapshot states; simultaneous selected target rows retained together; video16 has33 target rows in32 states',
            'future_utility_used_in_selection':False}
        for p in [directory/'SCAN_RESULT.json',directory/'START_MANIFEST.json',directory/'NATIVE_RNG_PROVENANCE.json',directory/'natural_event_index.jsonl.gz',directory/'hard_event_index.jsonl.gz',directory/'CONFLICT_GROUP_INDEX.json',Path(x['factual']['predictions'])]+[Path(s['path'])for s in x['bounded_snapshots']]:
            report['raw_manifest'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'published_payload':False})
    for role in('train','validation'):
        vs=[x for x in report['videos'].values()if x['role']==role]
        report[role]={'videos':len(vs),'MATCH_events':sum(x['counts']['MATCH_events']for x in vs),
            'Oracle2_wrong_proposal_correct_candidate_events':sum(x['counts']['strict_wrong_with_feasible_correct_candidate']for x in vs),
            'bounded_selected_events':sum(x['selected_events']for x in vs),'conflict_groups':sum(x['conflict_connected_groups']for x in vs),
            'video_units':len(vs),'scene_family_units':2,'verified_Oracle3_Oracle4_events':'PENDING'}
    save(REPORTS/'CORRECTIVE_OPPORTUNITY_AUDIT.json',report)
    (ROOT/'docs/PHASE8_CORRECTIVE_OPPORTUNITY_REPORT.md').write_text('''# Phase VIII corrective opportunity audit

All preregistered new TRAIN-controller videos completed the full native GMT-OFF scan. TRAIN12/13/14/16: 122,211 MATCH events and 2,278 prefix-anchored wrong proposals with a mathematically feasible correct candidate. Validation17/18/19: 38,887 MATCH events and 118 such events. These are Oracle1/2 availability bounds, not native corrective success or causal learning labels.

The original strict-purity anchor censors later contaminated histories. Before inspecting these new results, the immutable anchor amendment fixed historical identity at the first two consistent known observations; confirmed identities never rename after a wrong write. Both audit versions are preserved. All four new training videos and diagnostic video07 have byte-identical v0/v1 factual predictions. This observer change did not change the tracker. Unconfirmed IDs and duplicate aliases remain explicit. See PHASE8_ANCHOR_CONTRACT_AMENDMENT.md.

Chronological representative selection saved 82 training and 13 validation target events. There are 78 training and13 validation snapshot states; simultaneous target rows share the same state. The implementation caps snapshot states rather than separately counting simultaneous target groups, disclosed in the machine audit. Actual conflict-connected groups, group weights/Kish ESS, video and scene support are reported separately; consecutive errors cannot be independent samples. Candidate identities shared across targets connect conflict groups as well.

Every video passes 32 observational calls to the actual production GTR MATCH operator: exact OFF existing IDs, zero error in actual64-state inputs, unchanged complete mutable state. This proves the existing MATCH operator scope only. Direct candidate submission, native committed effects and H32 benefit still require separately scoped tests. Full deployed sliding-inference candidate integration is not verified.

Oracle3/4 is restricted to these sealed real states. CONTROL, explicit factual ACCEPT, native REASSOCIATE, semantic NEW with a same-key native-bank veto, and up to two real correct existing candidate alternatives share exact initial state and current raw scores/RNG. Future decisions use common live GMT OFF after actual mutations. H8/16/32 effects use frozen weights and birth-penalty-zero sensitivity, target errors and all-row externalities. No H32 expansion for every event, split reselection, Full24 or TEST is authorized.

The validation representative count is below the frozen20-event data gate even before filtering for causal benefit. It cannot be increased by counting neighbouring frames as independent. The native audit will establish the actual correction bound and whether current native interfaces, anchor uncertainty or effective support explain the failure. Raw matrices/snapshots/forks remain local; GitHub contains source, protocol, compact metrics and SHA manifests.
''')
    print(json.dumps({k:report[k]for k in('train','validation')}))
if __name__=='__main__':main()
