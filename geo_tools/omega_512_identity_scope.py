"""Check necessary native correction opportunities; no new GT matching, logits or solver replay."""
import json
from omega_common import ROOT,save_new,sha256


def main():
    e=json.loads((ROOT/'results/omega512_expansion_evidence_v1.json').read_text())
    assert e['manifest']['uniform_admission']['passes_pilot_gate']
    pilot=json.loads((ROOT/'results/omega512_independent_evidence_v1.json').read_text())
    passed=set(e['manifest']['summary']['C2_sift6000_mutual']['omega512']['pair_ids'])
    passed.update(pilot['manifest']['summary']['C2_sift6000_mutual']['omega512']['pair_ids'])
    cohort=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    pairs={p['pair_id']:p for p in cohort['pairs']}
    scope_path=ROOT/'results/omega_native_synchronous_scope_v1.json'
    scope=json.loads(scope_path.read_text());errors=scope['errors']
    rows=[]
    for error in errors:
        candidates=[p for p in pairs.values() if error['key'][0]==p['scene'] and error['key'][1] in {i['image_id'] for i in p['images']}]
        if not candidates:continue
        assert len(candidates)==1
        pair=candidates[0]
        other_ids={i['image_id'] for i in pair['images'] if i['camera_id']!=error['camera']}
        legal=[r for r in error['synchronous_references'] if r['source'][0] in other_ids]
        positive=sorted(set(r['track'] for r in legal)&set(error['synchronous_pure_positive_tracks']))
        rows.append(dict(error=error,pair_id=pair['pair_id'],sampling=pair['sampling'],
                         C2_joint_geometry_reliable=pair['pair_id'] in passed,
                         actual_legal_synchronous_references=legal,known_sync_positive_tracks=positive,
                         eligible_certified_correction_opportunity=pair['pair_id'] in passed and bool(positive)))
    eligible=[r for r in rows if r['eligible_certified_correction_opportunity']]
    scenes=sorted({r['error']['key'][0] for r in eligible})
    people=sorted({(r['error']['key'][0],r['error']['query_gt_identity']) for r in eligible})
    result=dict(schema='omega512-native-identity-scope/v1',status='NECESSARY_OPPORTUNITY_PRESENT' if len(scenes)>=2 and len(people)>=2 else 'INSUFFICIENT_CERTIFIED_MULTISCENE_OPPORTUNITY_IN_FIXED_COHORT',
                scope_source=dict(path=str(scope_path),sha256=sha256(scope_path)),
                reliable_pairs=sorted(passed),fixed_cohort_known_primary_errors=len(rows),
                eligible_certified_correction_opportunities=len(eligible),eligible_scenes=scenes,eligible_scene_qualified_people=people,
                rows=rows,no_new_GT_matching=True,no_GMT_inference=True,no_model_calls=True,
                actual_corrections='NOT_MEASURED',actual_harms='NOT_MEASURED',actual_joint_solver_changes='NOT_EXECUTED',
                interpretation='Necessary opportunities only, never evidence of net correction; missed/unknown/mixed histories retained.')
    save_new(ROOT/'results/omega512_identity_scope_v1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
