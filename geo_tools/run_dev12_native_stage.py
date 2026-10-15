"""Finite isolated GPU5 exporter controller; append-only independent ledger."""
import argparse
import time
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['off40_v1','on40_v1','full1200_v1'],required=True);a=p.parse_args()
    base=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');pre=base/'dev12_native_precheck_v1';pre.mkdir(exist_ok=True)
    scope=ROOT/'manifests/trackvggt_dev12_native_asset_release_v1.json';s=json.loads(scope.read_text());ledger=pre/'resource_ledger.json'
    report=json.loads(ledger.read_text()) if ledger.exists() else dict(id='dev12_native_v1',scope_sha256=sha256(scope),maximum_gpu_wall_hours=4.,runs=[])
    if report['scope_sha256']!=sha256(scope):raise ValueError('scope changed; preserve old ledger')
    if any(r['name']==a.stage for r in report['runs']):raise ValueError('stage already recorded; do not overwrite/restart')
    used=sum(r.get('outer_seconds',0) for r in report['runs']);remaining=4*3600-used
    if remaining<=0:raise ValueError('independent budget exhausted')
    # Physical GPU5 only; no overlap with the separately registered encoder
    # queues or another researcher. Query its actual UUID/process occupancy.
    uuid=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):raise ValueError('GPU5 occupied; do not displace any process')
    full=a.stage=='full1200_v1';output=base/'cv_pil_dev12_native_release_v1' if full else pre/a.stage
    command=[sys.executable,'geo_tools/export_gmt_dev12_native_assets.py','--scope',str(scope),'--output',str(output),'--groups','1200' if full else '40','--export','off' if a.stage=='off40_v1' else 'on']
    log=base/'logs'/f'dev12_native_{a.stage}.log'
    row=dict(name=a.stage,gpu=5,source_commit=SOURCE_COMMIT,command=command,output=str(output),log=str(log),status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['runs'].append(row);save_json(ledger,report)
    start=time.perf_counter();code=None
    try:
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save_json(ledger,report)
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30);raise ValueError('independent GPU wall budget exceeded')
    finally:
        row.update(outer_seconds=time.perf_counter()-start,status='PASS' if code==0 else 'FAILED',exit_code=code,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha256(log) if log.exists() else None)
        report['total_gpu_wall_hours']=sum(r.get('outer_seconds',0) for r in report['runs'])/3600;save_json(ledger,report)
    print(json.dumps(row))
    if code:raise SystemExit(code)


if __name__=='__main__':main()
