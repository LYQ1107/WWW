"""Snapshot only actual completed teacher reports; never impute missing scores."""
import csv
import torch
from common import *
from run_research_queue import process_identity


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[]
    for path in sorted((runtime/'models').glob('*/manifest.json')):
        value=json.loads(path.read_text())
        temporal=value.get('status')=='COMPLETE_MATCHED_REAL_CYCLE_CONTINUATION'
        if value.get('status')!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION' and not temporal:continue
        for name in ('checkpoint.pt','validation.json','validation_query_shuffle.json','loss_curve.json'):
            if sha256(path.parent/name)!=value['artifacts'][name]:raise ValueError('completed teacher artifact changed')
        if temporal:
            state=torch.load(path.parent/'checkpoint.pt',map_location='cpu');source=json.loads((Path(value['source_checkpoint']).parent/'manifest.json').read_text())
            value=dict(value,settings=state['settings'],all_fit_queries=source['all_fit_queries'],validation_queries=value['validation_metrics']['requests'],
                completed_steps=value['total_updates'],trainable_parameters=source['trainable_parameters'])
        settings=value['settings'];metrics=value['validation_metrics'];shuffle=value['query_shuffle_metrics']
        rows.append(dict(tag=path.parent.name,status=value['status'],fit_queries=value['all_fit_queries'],supervised_fit_queries=value['supervised_fit_queries'],
            validation_queries=value['validation_queries'],training_seed=settings['training_seed'],native_inference_seed=settings['actual_native_inference_seed'],
            raw_generic_provenance_seed=value['provenance']['seed'],seed_reporting_correction_required=value['provenance']['seed']!=settings['training_seed'],
            completed_updates=value['completed_steps'],backbone=settings['backbone'],training_phase='temporal_continuation' if temporal else 'no_cycle_fit',lambda_cycle=value.get('lambda_cycle',''),
            architecture=settings['architecture'],AG=settings['ag_mode'],QD=settings['qd_mode'] if settings['architecture']=='qd' else '',history_summary=settings['summary'],
            K=settings['k'],feature_variant=settings.get('feature_variant','full'),feature_plan_sha256=settings.get('feature_plan_sha256') or '',
            parameters=value['trainable_parameters'],Recall1=metrics['Recall1'],Recall5=metrics['Recall5'],MRR=metrics['MRR'],pair_AUC=metrics['pair_auc'],pair_AP=metrics['pair_ap'],
            matched_positive_denominator=metrics['matched_eligible'],known_queries=metrics['known_queries'],neutral_queries=metrics['neutral_small_target_queries'],
            shuffle_Recall1=shuffle['Recall1'],neutral_fit_forwards=value.get('neutral_fit_query_forwards',0),
            zero_gradient_batches=value.get('zero_gradient_batches',0),checkpoint_sha256=value['checkpoint_sha256'],manifest_sha256=sha256(path),manifest=str(path)))
    target=ROOT/'results/teacher_ablation.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    queues=[]
    for pattern in ['trackvggt_qd_fit_v1_gpu*','trackvggt_crop_fit_v1_gpu*','trackvggt_cycle_fit_v1_gpu*','trackvggt_qd_native_control_v1_gpu*','trackvggt_live_baseline_v1_gpu*']:
        for path in sorted((runtime/'queues').glob(pattern+'/manifest.json')):
            value=json.loads(path.read_text());live=process_identity(value['pid'])
            queues.append(dict(path=str(path),status=value['status'],pid=value['pid'],live_controller=live is not None,
                actual_command=live[1] if live else None,last_stage=value['stages'][-1] if value['stages'] else None))
    snapshot=dict(status='ACTUAL_TEACHER_PROGRESS_SNAPSHOT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_reports=len(rows),
        csv_sha256=sha256(target),rows=rows,queues=queues,GT_unknown_not_known_wrong=True,
        scope='Every number from an actual completed source-bound run.150/1000/1200 rows are distinct budgets. Temporal rows resolve seed15 from actual code/source checkpoint; raw generic default14 is retained and flagged. Pair results do not prove actual Hungarian/final tracking gain.',provenance=provenance())
    save_json(ROOT/'results/trackvggt_teacher_execution_snapshot_v1.json',snapshot)
    print(json.dumps(dict(status=snapshot['status'],completed_reports=len(rows),csv_sha256=sha256(target))))


if __name__=='__main__':main()
