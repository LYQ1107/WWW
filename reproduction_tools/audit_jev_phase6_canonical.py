"""Offline canonical provenance, input semantics and actual stale action-space audit."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from jev_phase6_common import OUT,REPORTS,sha,save,protect_anchor
from jev_phase6_offline_utility import OfflineIdentityAudit,prefix_identity
from gtr.modeling.jev_lifecycle import REACT_FIELDS,MEMORY_FIELDS

def action_space(video,canonical,actual):
    indexed=list(enumerate(actual.items()))
    ordered=sorted(indexed,key=lambda item:(item[1][0][:2],item[0]))
    cursor=0;counts=defaultdict(Counter);first={};events=[]
    for record in canonical:
        key=tuple(record['key'][1:3])
        while cursor<len(ordered) and ordered[cursor][1][0][:2]<key:
            index,(_,row)=ordered[cursor];cursor+=1
            if row['gt'] is None:continue
            track,target=row['id'],row['gt'];counts[track][target]+=1
            first[(track,target)]=min(first.get((track,target),index),index)
        mapping={t:max(c,key=lambda g:(c[g],-first[(t,g)])) for t,c in counts.items()}
        if len(events)<3 and mapping!=prefix_identity(actual,key)[0]:raise AssertionError('incremental offline identity mapping differs')
        context=record['online_context'];target=actual[(key[0],key[1],record['key'][4])]['gt']
        ranked=sorted(zip(context['candidate_track_ids'],context['candidate_scores']),key=lambda item:-item[1])
        valid=[(t,s) for t,s in ranked if s>context['bank_threshold']]
        assigned=record['assigned_track_id'];assigned_wrong=target is not None and assigned in mapping and mapping[assigned]!=target
        alternative=target is not None and any(t!=assigned and mapping.get(t)==target for t,s in valid)
        top=ranked[0][0] if ranked else None
        events.append({'key':record['key'],'GT':target,'assigned_id':assigned,'assigned_prefix_GT':mapping.get(assigned),
            'candidate_count':len(ranked),'valid_stale_count':len(valid),'assigned_wrong':assigned_wrong,
            'alternate_correct_for_assigned':alternative,'assigned_wrong_with_valid_correct_alternative':assigned_wrong and alternative,
            'top1_id':top,'top1_wrong':target is not None and top in mapping and mapping[top]!=target,
            'top2_correct':target is not None and len(ranked)>1 and mapping.get(ranked[1][0])==target and ranked[1][1]>context['bank_threshold'],
            'valid_candidates':[{'id':t,'score':s,'prefix_GT':mapping.get(t)} for t,s in valid]})
    assessed=sum(e['GT'] is not None and e['assigned_prefix_GT'] is not None for e in events)
    needed=sum(e['assigned_wrong_with_valid_correct_alternative'] for e in events)
    return {'events':events,'total':len(events),'assessable':assessed,'multiple_valid':sum(e['valid_stale_count']>=2 for e in events),
        'top1_wrong_top2_correct':sum(e['top1_wrong'] and e['top2_correct'] for e in events),
        'assigned_wrong_with_valid_correct_alternative':needed,'fraction_assessable_needing_alternative':needed/max(1,assessed),
        'third_action_data_gate':needed>=5 and needed/max(1,assessed)>=.1}

def main():
    data={};spaces={}
    for video in (24,23):
        root=OUT/f'lifecycle_native/video{video:02d}'
        manifest=json.loads((root/'CANONICAL_MANIFEST.json').read_text())
        if not manifest['native_MATCH_transition_contract']:raise AssertionError('legacy source excluded')
        journal=json.loads((root/'JOURNAL_RECONSTRUCTION.json').read_text())
        if journal['status']!='PASS' or not journal['all_factual_commits_identical']:raise AssertionError('native prefix failed')
        canonical=[json.loads(line) for line in (root/'canonical_reactivation.jsonl').open()]
        if any(r['GT_or_future_inputs'] or len(r['feature_vector'])!=len(REACT_FIELDS) or not all(math.isfinite(x) for x in r['feature_vector']) for r in canonical):raise AssertionError('relative online feature contract')
        gt=OfflineIdentityAudit(video);baseline=json.loads((root/'baseline/result.json').read_text())
        actual=gt.align(json.loads(Path(baseline['predictions']).read_text()))
        spaces[str(video)]=action_space(video,canonical,actual)
        labels=[json.loads(p.read_text()) for p in sorted((root/'labels').glob('REACT_*.json'))]
        data[str(video)]={'source_manifest':manifest,'source_manifest_sha256':sha(root/'CANONICAL_MANIFEST.json'),
            'journal_reconstruction_sha256':sha(root/'JOURNAL_RECONSTRUCTION.json'),
            'journal_fingerprint_scope':'nextID, active/stale/eligible IDs, ordered bank/raw memory/embeddings, hits, association assignments, RNG state/calls, relative metadata; diagnostic counters and immutable perceptions excluded',
            'canonical_relative_rows':len(canonical),'counterfactual_labels_so_far':len(labels),
            'counterfactual_informative_so_far':sum(r['sample_weight']>0 for r in labels),
            'control_exact_so_far':all(r['control_prefix_rng_and_native_bank_exact'] for r in labels)}
    complete=all(data[str(v)]['counterfactual_labels_so_far']==(32 if v==24 else 16) for v in (24,23))
    save(REPORTS/'REACTIVATION_CANONICAL_DATA_AUDIT.json',{'status':'COMPLETE' if complete else 'CANONICAL_COLLECTION_COMPLETE_LABELS_RUNNING',
        'videos':data,'coordinate_contract':'cache0 to annotation1, exact image IDs, one-to-one offline IoU>=0.5',
        'old_legacy_metadata_events_are_supervision':False,'legacy_commit_prefixes_used_in_training':False,
        'absolute_frame_or_virtual_frame_inputs':False,'future_policy':'frozen native B2 MATCH, live GMT MEMORY/REACT except isolated current intervention',
        'REACT_fields':REACT_FIELDS,'MEMORY_fields':MEMORY_FIELDS,'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Real canonical coverage differs from legacy metadata counts. Relative features and exact reconstructed native prefixes now provide actual train24/val23 events; coverage alone does not establish a useful learned recovery policy.'})
    save(REPORTS/'REACTIVATION_ACTION_SPACE_AUDIT.json',{'status':'COMPLETE','videos':spaces,
        'current_assignment':'full GMT stale-bank Hungarian proposal, which need not equal row top1',
        'third_action_rule':'at least5 assessable assigned-wrong events with a valid correct alternate, and >=10% of assessable events; necessary evidence only, not proof a constrained re-solve can assign that candidate',
        'third_action_eligible':any(s['third_action_data_gate'] for s in spaces.values()),
        'offline_GT_never_enters_runtime':True,'official_test_read':False,
        'what_did_we_learn':'Check alternatives against the actual constrained assignment as well as row ranking. The measured data gate decides whether a third stale action is justified; action count alone does not establish structured novelty.'})
    protect_anchor();print(json.dumps({'canonical':{v:{k:d[k] for k in ['canonical_relative_rows','counterfactual_labels_so_far']} for v,d in data.items()},
        'action_space':{v:{k:s[k] for k in ['multiple_valid','assigned_wrong_with_valid_correct_alternative','third_action_data_gate']} for v,s in spaces.items()}}))

if __name__=='__main__':main()
