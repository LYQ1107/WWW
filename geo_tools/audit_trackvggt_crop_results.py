"""All fixed224 resolution-repair results, with known/unknown paired outcomes."""
import collections
from common import *
from audit_trackvggt_ag_results import changes


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[];expected=None;fixes={}
    for backbone in ['vggt','dinov2']:
        for variant in ['full','local','combined']:
            root=runtime/'models'/f'ag_{backbone}_C_crop224_{variant}_k24_seed20261015_1000_v1'
            manifest=json.loads((root/'manifest.json').read_text())
            if manifest['status']!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION' or manifest['completed_steps']!=1000:raise ValueError('crop variant incomplete')
            for name,digest in manifest['artifacts'].items():
                if sha256(root/name)!=digest:raise ValueError('crop output changed')
            validation=json.loads((root/'validation.json').read_text());values=validation['rows'];signature={tuple(r['key']):[r[k] for k in ['query_gt_known','any_history_positive','top8_positive','known_unmatched','original_native_top1_state','actual_frozen_gmt_primary_state','patch_bin']] for r in values}
            if expected is None:expected=signature
            if signature!=expected or len(values)!=300:raise ValueError('fixed original validation inputs/labels differ')
            paired=changes(values);fixes[backbone,variant]={tuple(k) for k in paired['fix_keys']}
            bins={}
            for bucket in ['0','1-3','4-15','16+']:
                local=[r for r in values if r['patch_bin']==bucket];positive=[r for r in local if r['top8_positive']]
                bins[bucket]=dict(requests=len(local),matched_positive_queries=len(positive),Recall1=sum(r['positive_first_rank_including_null']==1 for r in positive)/len(positive) if positive else None,
                    states=dict(collections.Counter(r['state'] for r in local)),changes=changes(local))
            rows.append(dict(backbone=backbone,variant=variant,fit_queries=manifest['all_fit_queries'],supervised_fit_queries=manifest['supervised_fit_queries'],
                completed_updates=manifest['completed_steps'],parameters=manifest['trainable_parameters'],validation=manifest['validation_metrics'],query_shuffle=manifest['query_shuffle_metrics'],
                paired_changes=paired,original_size_bins=bins,neutral_fit_draws_in_last850=manifest['neutral_fit_query_forwards'],
                checkpoint_sha256=manifest['checkpoint_sha256'],manifest_sha256=sha256(root/'manifest.json')))
    comparisons=[dict(variant=v,VGGTranking_fix_keys=[list(k) for k in sorted(fixes['vggt',v])],DINOranking_fix_keys=[list(k) for k in sorted(fixes['dinov2',v])],
        VGGT_fix_keys_not_reproduced_by_matched_DINO=[list(k) for k in sorted(fixes['vggt',v]-fixes['dinov2',v])]) for v in ['full','local','combined']]
    summary=dict(status='COMPLETE_FIXED224_FULL_LOCAL_COMBINED_PRIMARY_COMPARISON',rows=rows,matched_DINO_fix_overlap=comparisons,
        actual_native_primary_known_error_queries_in_original300=sum(values[5] in ('wrong_link','missed_known_positive') for values in expected.values()),
        scope='Same fixed256 fit,234 GT-eligible candidate-backed queries, same seed/batches1000 updates and original300 validation (100 pure-positive ranking queries). No extended DEV12 input/model selection.',
        native_Recall1=.66,teacher_unique_benefit_established=False,
        limitation='Ranking fixes/harms are not actual solver/Bank/final identity changes. Full-image zero-center draws were not discarded; new valid crop support is a resolution mechanism, not unique VGGT geometry evidence.',provenance=provenance())
    save_json(ROOT/'results/trackvggt_crop_primary_results_v1.json',summary);print(json.dumps(dict(status=summary['status'],rows=[dict(backbone=r['backbone'],variant=r['variant'],Recall1=r['validation']['Recall1']) for r in rows])))


if __name__=='__main__':main()
