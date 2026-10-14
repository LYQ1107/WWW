"""Actual spawned VGGT camera worker; source boxes only, no label decisions."""
import argparse,time,numpy as np
from common import *
from gtr.geometry.process_worker import ProcessGeometryWorker,vggt_camera_factory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    source=json.loads((RUNTIME/'samples/screen_train_v1/samples.json').read_text())
    sample=source['samples'][0];arrays=np.load(sample['path'])
    records=[dict(file_name=v['path'],width=v['width'],height=v['height'],view_id=v['view_id']) for v in sample['views']]
    payload=dict(records=records,boxes=[arrays[f'boxes_{v}'].tolist() for v in range(2)])
    start=time.perf_counter();worker=ProcessGeometryWorker(vggt_camera_factory,dict(checkpoint=str(RUNTIME/'weights/vggt_1b.pt'),vggt_source=str(RUNTIME/'sources/vggt')),args.gpu,timeout=30.)
    try:
        if not worker.wait_ready(60.):raise RuntimeError(worker.error)
        startup=time.perf_counter()-start;submitted=time.monotonic()
        if not worker.submit(0,payload):raise AssertionError('initial task rejected')
        if worker.submit(1,payload):raise AssertionError('queue must remain bounded')
        result=None;deadline=time.monotonic()+30.
        while result is None and time.monotonic()<deadline:
            result=worker.poll(1)
            if worker.disabled:raise RuntimeError(worker.error)
            time.sleep(.01)
        if result is None:raise AssertionError('no worker response')
        if worker.encoder_calls!=1 or worker.stats['dropped_busy']!=1:raise AssertionError('worker counter mismatch')
        value=result['result']
        if value['valid'] and (value['F'] is None or not np.isfinite(value['F']).all()):raise AssertionError('invalid accepted F')
        summary=dict(source_frame=result['source_frame'],completed_after_submit=result['completed_at']>=submitted,completed_before_observation=result['completed_at']<=time.monotonic(),
            valid=value['valid'],validation_matches=value['validation_matches'],median_residual=value['median_residual'],peak_gpu_bytes=value['peak_gpu_bytes'])
    finally:shutdown=worker.close()
    report=dict(status='PASS',provenance=provenance(),worker_gpu=args.gpu,source_video_id=sample['video_id'],source_frame=sample['frame_id'],
        startup_seconds=startup,shutdown_seconds=shutdown,encoder_calls=worker.encoder_calls,worker_stats=worker.stats,result=summary,
        scope='actual public VGGT weights in a spawned separate CUDA process, CPU-only boxes/filenames IPC; source geometry validity, queue, completion and shutdown; no full tracker efficacy or speed claim')
    save_json(ROOT/f'results/{args.tag}.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
