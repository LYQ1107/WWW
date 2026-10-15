"""Native/Prior live replay with explicit setup/export and allocator peaks.

Delegates the unchanged causal actor/index to run_live_retrieval. Passive
wrappers observe mapping, synchronized tracker steps, and index queries.
Exact old standalone prefix/full predictions are a separate compulsory gate.
"""
import argparse
import collections
import resource
import shutil
import time
import numpy as np
import torch
from common import *


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--tag',required=True);p.add_argument('--backbone',default='none');p.add_argument('--max-frames',type=int);a,_=p.parse_known_args()
    forbidden={'--tracking-vggt-dense','--tracking-head-backbone','--diagnose-scoring'}
    if a.backbone!='none' or forbidden.intersection(sys.argv):raise ValueError('this registered baseline replay has no learned/foundation scorer')
    scope=json.loads((ROOT/'manifests/trackvggt_runtime_scope_v1.json').read_text());root=RUNTIME/'retrieval'/a.tag
    if scope['status']!='FROZEN_NATIVE_SYSTEM_BENCHMARK_SCOPE':raise ValueError('runtime scope not frozen')
    if shutil.disk_usage(RUNTIME).free<scope['minimum_free_disk_bytes']:raise ValueError('dedicated disk reserve violated')
    started=time.perf_counter()
    import run_live_retrieval as runner
    from gtr.geometry.isolated_stream import IsolatedCausalStream
    from gtr.geometry.retrieval_index import CausalClipIndex
    timers=collections.defaultdict(list);peaks=[];reserved=[]
    original_reset=torch.cuda.reset_peak_memory_stats;original_map=runner.map_records
    original_step=IsolatedCausalStream.step;original_query=CausalClipIndex.query
    def budget():
        limit=scope['maximum_prefix_process_seconds'] if a.max_frames else scope['maximum_full_process_seconds']
        cap=scope['maximum_prefix_output_bytes'] if a.max_frames else scope['maximum_full_output_bytes']
        if time.perf_counter()-started>limit:raise ValueError('finite live-replay wall budget exceeded; preserve partial failure')
        if sum(path.stat().st_size for path in root.rglob('*') if path.is_file())>cap:raise ValueError('finite live-replay storage budget exceeded')
    def reset(*args,**kwargs):
        peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
        return original_reset(*args,**kwargs)
    def mapped(*args,**kwargs):
        if len(timers['read_preprocess_mapping'])%100==0:budget()
        start=time.perf_counter();result=original_map(*args,**kwargs);timers['read_preprocess_mapping'].append(time.perf_counter()-start);return result
    def step(self,*args,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter();result=original_step(self,*args,**kwargs);torch.cuda.synchronize()
        timers['synchronized_tracker_step'].append(time.perf_counter()-start);return result
    def query(self,*args,**kwargs):
        start=time.perf_counter();result=original_query(self,*args,**kwargs);timers['index_query_compute'].append(time.perf_counter()-start);return result
    torch.cuda.reset_peak_memory_stats=reset;runner.map_records=mapped
    IsolatedCausalStream.step=step;CausalClipIndex.query=query
    try:runner.main()
    finally:
        torch.cuda.reset_peak_memory_stats=original_reset;runner.map_records=original_map
        IsolatedCausalStream.step=original_step;CausalClipIndex.query=original_query
    seconds=time.perf_counter()-started;peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
    budget();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['backbone']!='none' or manifest['tracking_method'] not in ('gmt_off','native_fixed_prior'):raise ValueError('baseline mode changed')
    if any(s['vggt_calls'] or s['dinov2_calls'] or s['failures'] for s in manifest['tracking_runtime_stats'].values()):raise ValueError('unexpected foundation call/failure in baseline')
    accounting=dict(status='COMPLETE_PROFILED_NATIVE_REPLAY',main_seconds_including_setup_export=seconds,
        process_import_and_final_accounting_scope='Parent queue outer monotonic wall time includes Python/import/setup/replay/export/accounting; use it for complete process FPS.',
        setup_seconds=manifest['setup_seconds_excluded'],loop_seconds=manifest['seconds'],
        outside_loop_after_main_setup_seconds=seconds-manifest['seconds']-manifest['setup_seconds_excluded'],
        camera_image_fps_including_main_setup_export=manifest['images']/seconds,
        components={k:dict(seconds=sum(v),calls=len(v),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95))) for k,v in timers.items()},
        allocator_peak_bytes_including_setup=max(peaks),allocator_peak_reserved_bytes_including_setup=max(reserved),
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU_device_name=torch.cuda.get_device_name(),CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        timing_effect='Explicit synchronized step profiling is charged to loop/process times equally for both methods; not an uninstrumented maximum-throughput or network-latency claim.',
        no_teacher_or_student_loaded=True,VGGT_calls=0,DINO_calls=0,GT_read_for_actor=False,
        limits='Read/preprocess combines file IO and deterministic transforms. Tracker includes detector/GTA/Bank/postprocessing. Query compute excludes outer response serialization; remaining loop includes publication/query serialization/logging/control.')
    save_json(root/'runtime_accounting.json',accounting)
    manifest.update(runtime_accounting=accounting,artifacts={**manifest['artifacts'],'runtime_accounting.json':sha256(root/'runtime_accounting.json')})
    save_json(root/'manifest.json',manifest)
    # Exact derivative receipt for the existing strict TrackEval evaluator;
    # never duplicate a multi-scene time as a per-scene speed measurement.
    if manifest['full_video']:
        for scene in manifest['video_ids']:
            prediction=root/f'video_{scene:02d}.json'
            receipt={k:manifest[k] for k in ('seed','checkpoint_sha256','config_sha256','annotation_sha256','commit','source_hashes','causal_online','full_video')}
            receipt.update(video_id=scene,predictions_sha256=sha256(prediction),parent_run_manifest=str(root/'manifest.json'),
                parent_run_manifest_sha256=sha256(root/'manifest.json'),derived_native_prediction_receipt=True,
                validation_native_output_prior=manifest['tracking_method']=='native_fixed_prior',geo_options=manifest['tracking_options'] or {},
                timing_scope='No per-scene FPS is inferred from shared multi-scene replay timing.')
            save_json(root/f'video_{scene:02d}_manifest.json',receipt)
    print(json.dumps(accounting))


if __name__=='__main__':main()
