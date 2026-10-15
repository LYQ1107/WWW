"""Paired query outcomes with explicit eligibility drift and fixes/harms."""
import argparse
from common import *


def compare(reference,candidate):
    if [row['query'] for row in reference]!=[row['query'] for row in candidate]:raise ValueError('detector query cohorts differ')
    if [row['gt_identity'] for row in reference]!=[row['gt_identity'] for row in candidate]:raise ValueError('query GT matching differs')
    def hit(row):return row['eligible'] and row['first_positive_rank']==1
    common=[(base,new) for base,new in zip(reference,candidate) if base['eligible'] and new['eligible']]
    fixes=sum(not hit(base) and hit(new) for base,new in common)
    harms=sum(hit(base) and not hit(new) for base,new in common)
    gained=sum(not base['eligible'] and new['eligible'] for base,new in zip(reference,candidate))
    lost=sum(base['eligible'] and not new['eligible'] for base,new in zip(reference,candidate))
    def reciprocal(row):return 1/row['first_positive_rank'] if row['first_positive_rank'] is not None else 0.
    changes=[]
    for base,new in zip(reference,candidate):
        if (base['eligible'],base['first_positive_rank'])!=(new['eligible'],new['first_positive_rank']):
            changes.append(dict(query=base['query'],baseline={key:base[key] for key in ('eligible','first_positive_rank','positive_gallery_clips','published_track_id')},
                method={key:new[key] for key in ('eligible','first_positive_rank','positive_gallery_clips','published_track_id')}))
    count=len(reference)
    return dict(requests=count,identical_detector_query_cohort=True,identical_query_gt_labels=True,
        baseline_eligible=sum(row['eligible'] for row in reference),method_eligible=sum(row['eligible'] for row in candidate),
        common_eligible=len(common),identical_eligible_query_set=not gained and not lost,
        eligible_gained=gained,eligible_lost=lost,common_eligible_rank1_fixes=fixes,common_eligible_rank1_harms=harms,
        common_eligible_rank1_delta=(fixes-harms)/len(common) if common else None,
        common_eligible_mrr_delta=sum(reciprocal(new)-reciprocal(base) for base,new in common)/len(common) if common else None,
        all_request_rank1_hit_delta=sum(hit(new)-hit(base) for base,new in zip(reference,candidate)),
        all_request_rank1_success_delta=sum(hit(new)-hit(base) for base,new in zip(reference,candidate))/count if count else None,
        changed_rank_or_eligibility_queries=len(changes),examples=changes[:50],examples_limit=50)


def shared_rank1_fixes(reference,vggt,control):
    compare(reference,vggt);compare(reference,control)
    def fixes(candidate):
        return {index for index,(base,new) in enumerate(zip(reference,candidate))
                if base['eligible'] and new['eligible'] and base['first_positive_rank']!=1 and new['first_positive_rank']==1}
    target,ordinary=fixes(vggt),fixes(control);shared=sorted(target&ordinary)
    return dict(vggt_rank1_fixes=len(target),control_rank1_fixes=len(ordinary),shared_rank1_fixes=len(shared),
        vggt_fixes_not_reproduced_by_this_control=len(target-ordinary),
        shared_fix_examples=[dict(query_ordinal=index,query=reference[index]['query'],baseline_rank=reference[index]['first_positive_rank'],
                                 vggt_rank=vggt[index]['first_positive_rank'],control_rank=control[index]['first_positive_rank']) for index in shared[:50]],
        scope='same ordered query occurrence, GT label, native baseline and stream-count comparison; identical aggregate gains alone do not establish shared corrections')


def load(path):
    value=json.loads(path.read_text())
    if value['status']!='MATCHES_RETAINED_METRICS' or not value['analysis_only'] or value['speed_eligible']:
        raise ValueError('verified post-run outcomes required')
    if sha256(value['metric'])!=value['metric_sha256'] or sha256(Path(value['run'])/'manifest.json')!=value['run_manifest_sha256']:
        raise ValueError('changed retained metric/run')
    metric=json.loads(Path(value['metric']).read_text());rows=value['outcomes']
    if len(rows)!=metric['requests'] or sum(row['matched'] for row in rows)!=metric['gt_matched_queries'] or sum(row['eligible'] for row in rows)!=metric['gallery_eligible_queries']:
        raise ValueError('outcome scope/counters differ')
    eligible=metric['gallery_eligible_queries']
    for row in rows:
        rank=row['first_positive_rank']
        if rank is not None and (not row['eligible'] or not isinstance(rank,int) or rank<1):raise ValueError('invalid outcome rank')
    if eligible:
        for k in [1,5,10]:
            recall=sum(row['first_positive_rank'] is not None and row['first_positive_rank']<=k for row in rows)/eligible
            if abs(recall-metric['recall_at_k'][str(k)])>1e-12:raise ValueError('outcome Recall differs from retained metric')
        mrr=sum(1/row['first_positive_rank'] for row in rows if row['first_positive_rank'] is not None)/eligible
        if abs(mrr-metric['mrr'])>1e-12:raise ValueError('outcome MRR differs from retained metric')
    for file,digest in value['raw_gt_sha256'].items():
        if sha256(file)!=digest:raise ValueError('raw GT changed')
    return rows,metric


def main():
    plan_path=ROOT/'manifests/retrieval_outcome_pairs_v1.json';plan=json.loads(plan_path.read_text())
    rows=[];pending=[];loaded_pairs={};artifacts={str(plan_path):sha256(plan_path)}
    for item in plan['pairs']:
        paths=[RUNTIME/'retrieval_outcomes'/item[key] for key in ('reference','candidate')]
        if any(not path.exists() for path in paths):pending.append(item);continue
        loaded=[load(path) for path in paths]
        result=compare(loaded[0][0],loaded[1][0])
        if any(row['query']['scene']!=item['query_scene'] for values,_ in loaded for row in values):raise ValueError('wrong query-scene scope')
        rows.append(dict(**item,**result,baseline_conditional_recall1=loaded[0][1]['recall_at_k']['1'],method_conditional_recall1=loaded[1][1]['recall_at_k']['1']))
        loaded_pairs[(item['method'],item['query_scene'])]=(loaded[0][0],loaded[1][0])
        artifacts.update({str(path):sha256(path) for path in paths})
    shared=[]
    for row in rows:
        if row['method'] not in {'VGGT2','VGGT4'}:continue
        baseline,vggt=loaded_pairs[(row['method'],row['query_scene'])]
        suffix=row['method'][-1]
        for control in ('GMT-head'+suffix,'DINO-head'+suffix):
            pair=loaded_pairs.get((control,row['query_scene']))
            if pair is None:continue
            if baseline!=pair[0]:raise ValueError('cross-control correction comparison needs identical native baseline outcomes')
            shared.append(dict(vggt_method=row['method'],control_method=control,query_scene=row['query_scene'],
                               **shared_rank1_fixes(baseline,vggt,pair[1])))
    save_json(ROOT/'results/retrieval_paired_query_outcomes_v1.json',dict(status='COMPLETE_DECLARED_COMPARISONS' if not pending else 'PARTIAL_VERIFIED_COMPARISONS',rows=rows,pending=pending,
        shared_control_corrections=shared,
        source_artifacts=artifacts,
        scope='exact same physical detector query cohort and matched GT identities; common eligible queries retain both corrections and harms; eligibility drift and all-request effects separate',
        limitations='GT post-run labels and original ranked responses only; no reranking, cached throughput, causal attribution to raw foundation vectors, stable seed effect or WWW deployment claim; query/eligible counts alone do not prove identical cohorts',provenance=provenance()))
    text=['# Paired detector-query retrieval outcomes','',
        'Each completed row verifies the exact physical query cohort and matched GT identities. Conditional Recall alone can hide corrections and harms or changed eligibility. Outcomes are recomputed from the retained original ranked responses after inference; old metric files are preserved. The recognized early four-counter schema is allowed only with explicit verification of every retained value.','',
        '| Method | Query scene | Requests | Common eligible | R1 fixes | R1 harms | Eligible gained/lost | Native R1 | Method R1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    number=lambda value:'missing' if value is None else f'{value:.6f}'
    for row in rows:
        text.append(f"| {row['method']} | {row['query_scene']} | {row['requests']} | {row['common_eligible']} | {row['common_eligible_rank1_fixes']} | {row['common_eligible_rank1_harms']} | {row['eligible_gained']}/{row['eligible_lost']} | {number(row['baseline_conditional_recall1'])} | {number(row['method_conditional_recall1'])} |")
    text+=['','Same-query correction overlap with completed matched-head controls:','',
        '| VGGT producer | Control | Scene | VGGT R1 fixes | Control R1 fixes | Shared fixes | VGGT fixes not reproduced by this control |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in shared:
        text.append(f"| {row['vggt_method']} | {row['control_method']} | {row['query_scene']} | {row['vggt_rank1_fixes']} | {row['control_rank1_fixes']} | {row['shared_rank1_fixes']} | {row['vggt_fixes_not_reproduced_by_this_control']} |")
    text+=['','Shared corrections compare original ordered query occurrences, not just aggregate counts; duplicate physical-query values are not collapsed. They are observational counterexamples to unique VGGT attribution for those corrected queries, not proof that all foundation features or geometries are useless.','']
    text+=['',f'Pending comparison artifacts: {len(pending)}. Their missing values are not zero. Full examples, cohort/metric hashes and MRR changes are in results/retrieval_paired_query_outcomes_v1.json.','',
        'These are query-rank changes, distinct from all-time tracking identity pairs or internal Hungarian decisions. A single primary-head local replay cannot establish stable VGGT-specific benefit, a speed advantage or WWW deployment.','']
    (ROOT/'docs/GEO_STREAM_RETRIEVAL_QUERY_AUDIT.md').write_text('\n'.join(text))
    print(json.dumps([dict(method=row['method'],scene=row['query_scene'],same_eligibility=row['identical_eligible_query_set'],fixes=row['common_eligible_rank1_fixes'],harms=row['common_eligible_rank1_harms'],eligible_gained=row['eligible_gained'],eligible_lost=row['eligible_lost']) for row in rows]))


if __name__=='__main__':main()
