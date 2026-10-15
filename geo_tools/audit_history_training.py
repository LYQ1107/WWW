"""Verify bounded history-training artifacts and report actual whole-command cost."""
import numpy as np
import torch
from common import *
from train_history_heads import history_clauses


def main():
    queue_path=RUNTIME/'queues/history_training_v1_gpu4/manifest.json';queue=json.loads(queue_path.read_text())
    if queue['status']!='FINISHED' or any(row['status']!='PASS' for row in queue['stages']):
        raise ValueError('all collector/trainer commands must complete first')
    path=RUNTIME/'matrix_reports/history_rank_three_backbones_train_v1.json';training=json.loads(path.read_text())
    if training['status']!='HISTORY_TRAIN_DIAGNOSTIC_COMPLETE':raise ValueError('training method did not complete')
    rows=[];error_contexts=[]
    for video in [5,8,12]:
        run=RUNTIME/'history_train'/f'history_train_keyframe_v1_v{video}';manifest=run/'manifest.json';info=json.loads(manifest.read_text())
        if not info['neutral_prefix40_exact'] or info['encoder_failures']:raise ValueError('collection correctness/failure gate')
        if training['source_manifests'][str(manifest)]!=sha256(manifest):raise ValueError('changed collection manifest')
        for name,value in info['artifacts'].items():
            if sha256(run/name)!=value:raise ValueError('changed training artifact')
        arrays=np.load(run/'targets.npz')
        if video==12:
            observations=json.loads((run/'observations.json').read_text())
            clauses=history_clauses(observations,arrays['labels'],torch.load(run/'events.pt',map_location='cpu'))
            for index,clause in enumerate(clauses):
                if clause['positive'][clause['base'].argmax()]:continue
                gap=float(clause['base'][~clause['positive']].max()-clause['base'][clause['positive']].max())
                error_contexts.append(dict(clause_index=index,query_observation=clause['query'],
                    query_image_id=observations[clause['query']]['image_id'],
                    native_negative_minus_positive_logit=gap,exceeds_maximum_pairwise_residual_difference4=gap>4.))
        rows.append(dict(video_id=video,observations=info['observations'],events=info['events'],
            known_gt_observations=int((arrays['labels']>=0).sum()),vector_dimensions={name:int(arrays[name].shape[1]) for name in ['gmt','dinov2','vggt']},
            vggt_calls=info['vggt_calls'],dinov2_calls=info['dinov2_calls'],neutral_prefix40_exact=info['neutral_prefix40_exact'],
            collection_loop_seconds=info['collection_seconds'],saved_artifact_bytes=sum((run/name).stat().st_size for name in info['artifacts']),
            manifest_sha256=sha256(manifest)))
    native=training['native'];n=len(native['correct'])
    if abs(native['recall1']-sum(native['correct'])/n)>1e-12:raise ValueError('native rank count mismatch')
    observed=set()
    for result in training['results']:
        observed.add((result['backbone'],result['seed']));head=Path(result['checkpoint'])
        if sha256(head)!=result['checkpoint_sha256']:raise ValueError('changed selected head')
        checkpoint=torch.load(head,map_location='cpu')
        if checkpoint['source_split']!='association_train' or checkpoint['seed']!=result['seed']:raise ValueError('head provenance mismatch')
        if checkpoint['feature_weight']!=2. or checkpoint['max_residual']!=2.:raise ValueError('unexpected residual bound')
        after=result['after'];correct=after['correct']
        if len(correct)!=n or abs(after['recall1']-sum(correct)/n)>1e-12:raise ValueError('head rank count mismatch')
        if after['fixes_native']!=sum(not a and b for a,b in zip(native['correct'],correct)) or after['harms_native']!=sum(a and not b for a,b in zip(native['correct'],correct)):
            raise ValueError('head correction/harm count mismatch')
    if len(training['results'])!=9 or observed!={(kind,seed) for kind in ['gmt','dinov2','vggt'] for seed in [20261014,20261015,20261016]}:
        raise ValueError('matched three-backbone/three-seed matrix incomplete')
    save_json(ROOT/'results/history_rank_train_audit_v1.json',dict(status='PASS_BOUNDED_TRAIN_DIAGNOSTIC',
        unique_vggt_primary_gate=training['unique_vggt_primary_gate'],native_rank_errors=n-sum(native['correct']),
        training_report=str(path),training_report_sha256=sha256(path),queue_manifest_sha256=sha256(queue_path),collectors=rows,
        actual_whole_command_seconds=sum(stage['seconds'] for stage in queue['stages']),
        native_error_contexts=error_contexts,
        native_error_unique_query_observations=len({row['query_observation'] for row in error_contexts}),
        error_bound_scope='fixed logged query/reference-frame logits: tanh evidence times2 and max-residual2 can change a positive-versus-negative gap by at most4. This is not a bound on a rerun with changed tracking history or whole GTA/Hungarian behavior',
        command_costs=[dict(id=stage['id'],seconds=stage['seconds'],log=stage['log'],log_sha256=stage['log_sha256']) for stage in queue['stages']],
        timing_scope='whole command durations include imports, model/data setup, inference, neutral-prefix recheck, labels/archive/head writes and report hashing; sum excludes the scheduler wait before this priority window. Individual collection_loop_seconds excludes GMT setup, the second OFF pass, labels and archive writes',
        scope='three fixed TRAIN prefixes; actual OFF40 raw-output parity before GT labels; matched9 head checkpoints; historical reference-frame ranking only, not DEV or full tracking',
        limitations='same heldout TRAIN12 reused for checkpoint selection; correlated clauses repeat queries across past frames; failed unique-feature gate is a bounded negative diagnostic, not proof all VGGT features are useless',provenance=provenance()))
    summary=dict(training,native={k:v for k,v in native.items() if k!='correct'},
        results=[dict(row,before={k:v for k,v in row['before'].items() if k!='correct'},
                      after={k:v for k,v in row['after'].items() if k!='correct'}) for row in training['results']],
        external_full_report=str(path),external_full_report_sha256=sha256(path),
        saved_summary_scope='per-clause correctness arrays remain in the external hashed full report; aggregate values independently verified by audit_history_training.py')
    save_json(ROOT/'results/history_rank_three_backbones_train_v1.json',summary)
    print(json.dumps(dict(gate=training['unique_vggt_primary_gate'],native_rank_errors=n-sum(native['correct']),heads=len(observed))))


if __name__=='__main__':main()
