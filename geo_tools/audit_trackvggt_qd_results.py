"""Ordinary/track QD comparisons and candidate-aligned query-shuffle effects."""
import numpy as np
from common import *
from audit_trackvggt_ag_results import changes


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[];pending=[];reference=None
    cases=[('vggt',m) for m in ['last','mean','concat','global','track']]+[('dinov2',m) for m in ['global','track']]
    for backbone,mode in cases:
        tag=f'qd_{backbone}_{mode}_k24_seed20261015_v3_1000_v1';root=runtime/'models'/tag;path=root/'manifest.json'
        if not path.exists():pending.append(tag);continue
        m=json.loads(path.read_text())
        if m['status']!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION':raise ValueError('invalid QD report')
        for name,digest in m['artifacts'].items():
            if sha256(root/name)!=digest:raise ValueError('fitted QD artifact changed')
        ordinary=json.loads((root/'validation.json').read_text())['rows'];shuffled=json.loads((root/'validation_query_shuffle.json').read_text())['rows']
        query_keys=[r['key'] for r in ordinary]
        if reference is None:reference=query_keys
        if query_keys!=reference or len(ordinary)!=300:raise ValueError('QD validation membership/order differs')
        distances=[];permuted_pairs=0;fixed_pairs=0;zero_weight_queries=0
        for before,after in zip(ordinary,shuffled):
            if before['key']!=after['key']:raise ValueError('shuffle changed physical query')
            weights=np.asarray(before['layer_weights']);changed=np.asarray(after['layer_weights']);permutation=after['query_permutation']
            fixed_pairs+=sum(i==j for i,j in enumerate(permutation or []));permuted_pairs+=sum(i!=j for i,j in enumerate(permutation or []))
            if not len(weights):zero_weight_queries+=1;continue
            if changed.shape!=weights.shape:raise ValueError('shuffle weight layout changed')
            if permutation is None or not np.allclose(changed,weights[permutation],atol=1e-7,rtol=1e-6):raise ValueError('layer weights are not computed from the actual permuted query')
            distances.extend(np.abs(weights-changed).sum(1).tolist())
        rows.append(dict(tag=tag,backbone=backbone,QD=mode,parameters=m['trainable_parameters'],validation=m['validation_metrics'],
            query_shuffle=m['query_shuffle_metrics'],paired_changes_vs_native=changes(ordinary),
            candidate_aligned_layer_weight_shuffle_L1_mean=float(np.mean(distances)) if distances else None,
            history_permutation_weight_alignment_exact_within_tolerance=True,fixed_candidate_pairs=fixed_pairs,permuted_candidate_pairs=permuted_pairs,
            queries_without_layer_weights=zero_weight_queries,checkpoint_sha256=m['checkpoint_sha256'],manifest_sha256=sha256(path)))
    control=runtime/'models/native_mlp_qd_capacity_seed20261015_v3_1000_v1/manifest.json'
    if not control.exists():pending.append('native_mlp_qd_capacity_seed20261015_v3_1000_v1');native=None
    else:
        m=json.loads(control.read_text())
        if sha256(control.parent/'checkpoint.pt')!=m['checkpoint_sha256']:raise ValueError('QD-capacity native control changed')
        native=dict(parameters=m['trainable_parameters'],validation=m['validation_metrics'],query_shuffle=m['query_shuffle_metrics'],checkpoint_sha256=m['checkpoint_sha256'])
    proof=dict(status='COMPLETE_MATCHED_QD_PRIMARY_COMPARISON' if not pending else 'PARTIAL_MATCHED_QD_PRIMARY_COMPARISON',rows=rows,pending=pending,native_QD_capacity_control=native,
        weights_interpretation='Mean/std over all candidate queries is invariant to permuting that same query multiset. Candidate-aligned changes and exact permutation alignment test actual query conditioning; a change alone does not establish useful identity information.',
        shuffle_scope='Only the actual history query is permuted. Current image, legal candidate order and native per-candidate pair evidence remain fixed. Fixed permutation points and neutral queries are disclosed.',
        native_Recall1=.66,scientific_success=False,full_tracking_success=False,provenance=provenance())
    save_json(ROOT/'results/trackvggt_qd_primary_results_v1.json',proof);print(json.dumps(dict(status=proof['status'],completed=len(rows),pending=len(pending))))


if __name__=='__main__':main()
