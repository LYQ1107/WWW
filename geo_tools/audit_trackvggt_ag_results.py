"""Audit completed V2/V3 AG results; ranking fixes are not solver repairs."""
import argparse
import collections
from common import *


def changes(rows):
    def correct(state):return state in ('correct_link','correct_null_within_recorded_history')
    def wrong(state):return state in ('wrong_link','missed_known_positive')
    fixes=[r['key'] for r in rows if wrong(r['original_native_top1_state']) and correct(r['state'])]
    harms=[r['key'] for r in rows if correct(r['original_native_top1_state']) and wrong(r['state'])]
    uncertain=[r['key'] for r in rows if correct(r['original_native_top1_state']) and not correct(r['state']) and not wrong(r['state'])]
    return dict(known_ranking_fixes=len(fixes),known_ranking_harms=len(harms),native_correct_to_uncertified=len(uncertain),
        fix_keys=fixes,harm_keys=harms,uncertified_keys=uncertain,
        scope='Original native Top1 vs learned Top1 ranking, not actual joint Hungarian/Bank/final tracking corrections. Unknown/ambiguous are never known wrong.')


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');required=[];queues=[]
    for plan in ['trackvggt_phase2_ag_fit_v1','trackvggt_v3_ag_fit_v1']:
        for gpu in [0,4,5,6]:
            path=runtime/'queues'/f'{plan}_gpu{gpu}/manifest.json';q=json.loads(path.read_text())
            if q['status']!='FINISHED' or any(s['status']!='PASS' for s in q['stages']):raise ValueError('original AG comparison queue not complete')
            queues.append(dict(plan=plan,gpu=gpu,path=str(path),sha256=sha256(path),outer_seconds=sum(s['seconds'] for s in q['stages'])))
            required.extend(s['id'] for s in q['stages'] if ('1000' in s['id']))
    summaries=[];expected_keys=None
    for tag in required:
        root=runtime/'models'/tag;manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in manifest['artifacts'].items():
            if sha256(root/name)!=digest:raise ValueError('completed teacher artifact changed')
        validation=json.loads((root/'validation.json').read_text());rows=validation['rows'];keys={tuple(r['key']) for r in rows}
        if expected_keys is None:expected_keys=keys
        if keys!=expected_keys or len(rows)!=300:raise ValueError('original validation300 differs')
        bins={}
        for bucket in ['0','1-3','4-15','16+']:
            local=[r for r in rows if r['patch_bin']==bucket];eligible=[r for r in local if r['top8_positive']]
            bins[bucket]=dict(requests=len(local),pure_positive_queries=len(eligible),
                pure_positive_Recall1=sum(r['positive_first_rank_including_null']==1 for r in eligible)/len(eligible) if eligible else None,
                predicted_states=dict(collections.Counter(r['state'] for r in local)),changes=changes(local))
        summaries.append(dict(tag=tag,settings=manifest['settings'],completed_updates=manifest['completed_steps'],
            fit_queries=manifest['all_fit_queries'],supervised_fit_queries=manifest['supervised_fit_queries'],parameters=manifest['trainable_parameters'],
            validation=manifest['validation_metrics'],query_shuffle=manifest['query_shuffle_metrics'],paired_changes=changes(rows),bins=bins,
            model_manifest_sha256=sha256(root/'manifest.json'),checkpoint_sha256=manifest['checkpoint_sha256']))
    proof=dict(status='COMPLETE_PRIMARY_V2_V3_AG_COMPARISON',completed_full_settings=len(summaries),queues=queues,rows=summaries,
        original_validation_queries=300,original_positive_ranking_denominator=100,native_Recall1=.66,NativePrior_Recall1=.66,
        every_candidate_GT_unknown_preserved=True,teacher_unique_benefit=False,
        scope='Primary seed20261015, fitted1000 updates, frozen target pool and native history. V2/V3 compare same architecture/update budget. Crops/QD/cycle are separate uncompleted hypotheses.',provenance=provenance())
    save_json(ROOT/'results/trackvggt_ag_v2_v3_primary_results_v1.json',proof)
    print(json.dumps(dict(status=proof['status'],rows=[dict(tag=r['tag'],Recall1=r['validation']['Recall1'],shuffle=r['query_shuffle']['Recall1'],
        known_fixes=r['paired_changes']['known_ranking_fixes'],known_harms=r['paired_changes']['known_ranking_harms']) for r in summaries])))


if __name__=='__main__':main()
