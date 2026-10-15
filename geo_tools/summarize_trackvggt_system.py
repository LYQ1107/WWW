"""Publish only completed, parity-verified fresh native replay measurements.

Missing cases remain explicit rows with blank measurements. Inherited full MOT
references are separate rows and never stand in for fresh timing or students.
"""
import csv
from common import *


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    runtime = Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
    timing, tracking, receipts = [], [], []
    gate_path=runtime/'matrix_reports/trackvggt_runtime_prefix_gate_v1.json'
    gate=None
    if gate_path.exists():
        value=json.loads(gate_path.read_text())
        if value['status']!='PASS' or value['full_video'] or len(value['rows'])!=2:
            raise ValueError('fresh native prefix gate has not passed at its stated scope')
        gate=dict(path=str(gate_path),sha256=sha256(gate_path),status=value['status'])
        (ROOT/'results/trackvggt_runtime_prefix_gate_v1.json').write_bytes(gate_path.read_bytes())
    for streams, gpu in [(2, 'gpu0'), (4, 'gpu4'), (10, 'gpu6')]:
        queue_path = runtime/'queues'/f'trackvggt_live_baseline_v1_{gpu}'/'manifest.json'
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
        for short, method in [('off', 'gmt_off'), ('prior', 'native_fixed_prior')]:
            tag = f'trackvggt_runtime_{short}_{streams}streams_full_v1'
            root = runtime/'retrieval'/tag
            row = dict(origin='FRESH_PROFILED_NATIVE_SYSTEM_REPLAY', method=method,
                       configured_logical_streams=streams, inference_seed=20261014,
                       status='PENDING_REAL_REPLAY_AND_PARITY', tag=tag)
            stage = next((s for s in queue.get('stages', []) if s['id']==tag), None)
            if stage:
                row['status'] = stage['status']
            parity_path = runtime/'matrix_reports'/(tag+'_parity.json')
            manifest_path = root/'manifest.json'
            if not (stage and stage['status']=='PASS' and parity_path.exists() and manifest_path.exists()):
                timing.append(row)
                continue
            manifest = json.loads(manifest_path.read_text())
            parity = json.loads(parity_path.read_text())
            if parity['status']!='PASS' or not manifest['full_video']:
                raise ValueError('completed system replay lacks valid full parity')
            if manifest['tracking_method']!=method or manifest['camera_streams']!=streams:
                raise ValueError('system method/stream scope differs')
            if manifest['backbone']!='none' or manifest.get('head_checkpoint_sha256') or manifest.get('tracking_head_checkpoint_sha256'):
                raise ValueError('native benchmark unexpectedly loaded learned scoring')
            for name, digest in manifest['artifacts'].items():
                if sha256(root/name)!=digest:
                    raise ValueError('completed system artifact changed: '+name)
            if sha256(stage['log'])!=stage['log_sha256']:
                raise ValueError('completed system process log changed')
            accounting = json.loads((root/'runtime_accounting.json').read_text())
            if accounting['status']!='COMPLETE_PROFILED_NATIVE_REPLAY':
                raise ValueError('missing complete cost accounting')
            if not stage['seconds']>=accounting['main_seconds_including_setup_export']>=manifest['seconds']+accounting['setup_seconds']:
                raise ValueError('complete process/setup/loop timing scopes are inconsistent')
            row.update(status='PASS_FULL_REPLAY_AND_PREDICTION_PARITY', images=manifest['images'],
                complete_process_seconds=stage['seconds'], complete_process_camera_image_fps=manifest['images']/stage['seconds'],
                main_setup_export_seconds=accounting['main_seconds_including_setup_export'],
                replay_loop_seconds=manifest['seconds'], replay_loop_camera_image_fps=manifest['camera_image_fps'],
                setup_seconds=accounting['setup_seconds'],
                GPU_device_name=accounting['GPU_device_name'],CUDA_visible_devices=accounting['CUDA_visible_devices'],
                read_preprocess_seconds=accounting['components']['read_preprocess_mapping']['seconds'],
                synchronized_tracker_seconds=accounting['components']['synchronized_tracker_step']['seconds'],
                index_query_compute_seconds=accounting['components'].get('index_query_compute', {}).get('seconds'),
                round_p50_seconds=manifest['round_latency_seconds']['p50'],round_p95_seconds=manifest['round_latency_seconds']['p95'],
                query_response_p50_seconds=manifest['query_latency_seconds']['p50'],query_response_p95_seconds=manifest['query_latency_seconds']['p95'],
                allocator_peak_bytes_including_setup=accounting['allocator_peak_bytes_including_setup'],
                host_peak_rss_bytes=accounting['host_peak_rss_bytes'],index_peak_payload_bytes=manifest['index']['peak_payload_bytes'],
                VGGT_calls=accounting['VGGT_calls'],DINO_calls=accounting['DINO_calls'],manifest_sha256=sha256(manifest_path))
            components=accounting['components']
            row['remaining_replay_loop_seconds_including_publication_and_control']=manifest['seconds']-sum(c['seconds'] for c in components.values())
            row['synchronized_tracker_step_p50_seconds']=components['synchronized_tracker_step']['p50']
            row['synchronized_tracker_step_p95_seconds']=components['synchronized_tracker_step']['p95']
            metric_path = root/'retrieval.metrics.json'
            if metric_path.exists():
                metric = json.loads(metric_path.read_text())
                if metric['status']!='PASS' or metric['run_manifest_sha256']!=sha256(manifest_path):
                    raise ValueError('retrieval metric not bound to this run')
                for path, digest in metric['raw_gt_sha256'].items():
                    if sha256(path)!=digest:raise ValueError('retrieval raw GT changed')
                row.update(retrieval_requests=metric['requests'],GT_matched_queries=metric['gt_matched_queries'],
                    gallery_eligible_queries=metric['gallery_eligible_queries'],detector_query_precision=metric['detector_query_precision'],
                    GT_query_detection_coverage=metric['gt_query_detection_coverage'],
                    retrieval_Recall1=metric['recall_at_k']['1'],retrieval_Recall5=metric['recall_at_k']['5'],
                    retrieval_Recall10=metric['recall_at_k']['10'],retrieval_MRR=metric['mrr'],
                    retrieval_success1_all_requests=metric['success_at_k_all_requests']['1'],
                    retrieval_metric_sha256=sha256(metric_path))
            timing.append(row)
            receipts.append(dict(tag=tag,manifest=str(manifest_path),sha256=sha256(manifest_path),
                parity=str(parity_path),parity_sha256=sha256(parity_path),queue=str(queue_path),queue_sha256=sha256(queue_path)))
    inherited=json.loads((ROOT/'results/trackvggt_inherited_native_reference_v1.json').read_text())
    for row in inherited['rows']:
        path=Path(row['metric_file'])
        if sha256(path)!=row['metric_sha256']:raise ValueError('inherited metric receipt changed')
        tracking.append(dict(origin=row['origin'],method=row['method'],inference_seed=row['inference_seed'],
            status='VERIFIED_INHERITED_FULL_DEV5',images=row['images'],**row['metrics'],metric_sha256=row['metric_sha256']))
    for short,method in [('off','gmt_off'),('prior','native_fixed_prior')]:
        tag=f'trackvggt_runtime_{short}_10streams_full_v1'
        path=runtime/'matrix_reports'/(tag+'_MOT_metrics.json')
        row=dict(origin='FRESH_PROFILED_NATIVE_SYSTEM_REPLAY',method=method,inference_seed=20261014,status='PENDING_FRESH_FULL_DEV5_METRICS')
        if path.exists():
            metrics=json.loads(path.read_text())
            if metrics['status']!='PASS' or not metrics['full_video'] or metrics['video_ids']!=[1,10,11,18,19]:
                raise ValueError('fresh MOT metric scope differs')
            for prediction,digest in metrics['prediction_sha256'].items():
                if sha256(prediction)!=digest:raise ValueError('fresh full predictions changed')
            for truth,digest in metrics['raw_gt_sha256'].items():
                if sha256(truth)!=digest:raise ValueError('fresh MOT GT changed')
            raw=runtime/'suites'/(tag+'_strict5')/'per_view/combined_raw_metrics.json'
            clear=json.loads(raw.read_text())['CLEAR']
            row.update(status='PASS_FRESH_FULL_DEV5',images=metrics['images'],**metrics['metrics'],
                Frag=int(clear['Frag']),metric_sha256=sha256(path),raw_combined_metric_sha256=sha256(raw))
        tracking.append(row)
    write_csv(ROOT/'results/runtime_comparison.csv',timing)
    write_csv(ROOT/'results/full_tracking_metrics.csv',tracking)
    pending=[r['tag'] for r in timing if r['status']!='PASS_FULL_REPLAY_AND_PREDICTION_PARITY' or 'retrieval_Recall1' not in r]
    pending_tracking=[r['method'] for r in tracking if r['status']=='PENDING_FRESH_FULL_DEV5_METRICS']
    summary=dict(status='COMPLETE_FRESH_NATIVE_SYSTEM_EVALUATION' if gate and not pending and not pending_tracking else 'ACTUAL_NATIVE_SYSTEM_PROGRESS_SNAPSHOT',
        rows=timing,tracking_rows=tracking,receipts=receipts,prefix_gate=gate,pending=pending,pending_tracking_metrics=pending_tracking,
        timing_scope='Complete process FPS uses actual queue outer wall time. Profiled local file replay, unequal-length independent two-camera videos; configured streams are not guaranteed constant active membership. Query response latency excludes network.',
        learned_system_results='NOT_RUN_NO_VALID_STUDENT_OR_TEACHER_DEPLOYMENT',scientific_success=False,provenance=provenance())
    save_json(ROOT/'results/trackvggt_system_execution_snapshot_v1.json',summary)
    print(json.dumps(dict(status=summary['status'],pending=len(summary['pending']))))


if __name__=='__main__':main()
