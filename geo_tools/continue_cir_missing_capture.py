"""Complete only the12 missing native input stores after DEV36 releases GPUs.

No identity-error mining, GT parsing, new video download or learned model. The
existing Original predictions and input images bind every collected scene.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, save_json, sha256
from run_cir_guarded import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--root-ledger',type=Path,required=True);p.add_argument('--prefix-ledger',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);rt=a.runtime;start=time.monotonic()
    state=dict(status='WAIT_ACTIVE_DEV36',pid=os.getpid(),scientific_success=False);save_json(a.out/'STATE.json',state)
    while True:
        ledger=json.loads(a.root_ledger.read_text());running=[s for s in ledger['stages'] if s['status'].startswith('RUNNING')]
        for s in running:
            if not Path('/proc/'+str(s['supervisor_pid'])).exists():raise RuntimeError('Missing current resource supervisor; no restart')
        memory=int(subprocess.check_output(['nvidia-smi','-i','9','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        state.update(wait_seconds=time.monotonic()-start,active=[s['stage'] for s in running]);save_json(a.out/'STATE.json',state)
        if not running and not memory:break
        if time.monotonic()-start>21660:raise TimeoutError('Bounded wait expired')
        time.sleep(15)
    protocol=json.loads((ROOT/'results/CIR_GMT/MISSING_NATIVE_CAPTURE_PROTOCOL.json').read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Clean frozen collector source required')
    prefix_root=rt/'missing_capture_prefix_v3';prefix_out=prefix_root/'scene1'
    common=['--reference-runs',protocol['reference_runs'],'--reference-sha256',protocol['reference_runs_sha256']]
    state['status']='PREFIX_V3_RUNNING';save_json(a.out/'STATE.json',state)
    result=subprocess.run([sys.executable,'geo_tools/run_cir_guarded.py','--ledger',str(a.prefix_ledger),
        '--stage','P5_MISSING_NATIVE_REAL40_CAPTURE_V3','--gpu','9','--seconds','300',
        '--log',str(a.out/'prefix.log'),'--cwd',str(ROOT),'--source-commit',head,
        '--',sys.executable,'geo_tools/capture_cir_missing_native.py','--video','1',*common,
        '--prefix40','--stage-root',str(prefix_root),'--out',str(prefix_out)],cwd=ROOT)
    if result.returncode:
        state.update(status='PREFIX_FAILED_NO_FULL_CAPTURE_ADMISSION',exit_code=result.returncode);save_json(a.out/'STATE.json',state);return
    prefix=json.loads((prefix_out/'manifest.json').read_text())
    if prefix['status']!='PASS_REAL40_MISSING_INPUT_CAPTURE' or not prefix['predictions_exact_original']:
        raise ValueError('Actual prefix input/output contract not qualified')
    # Do not duplicate a newly available peer feature export. Metadata only;
    # no peer GT labels, retained error examples or learned outputs are read.
    peer=Path('/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_audit_v1')
    peer_meta=[]
    for video in protocol['missing_native_videos']:
        path=peer/f'scene{video}_full_v1/manifest.json'
        if path.exists():
            meta=json.loads(path.read_text());vectors=meta.get('persistent_native_feature_vectors',0)
            peer_meta.append(dict(video=video,status=meta.get('status'),persistent_native_feature_vectors=vectors))
            if vectors:raise RuntimeError('Peer now retains native vectors; admit reusable release before any duplicate capture')
    save_json(a.out/'PEER_METADATA_RECHECK.json',peer_meta)
    subprocess.run([sys.executable,'geo_tools/merge_cir_subledger.py','--parent',str(a.root_ledger),
        '--child',str(a.prefix_ledger)],cwd=ROOT,check=True)
    destination=rt/'missing_native_full12_v1';destination.mkdir(exist_ok=False);jobs=[]
    for video in protocol['missing_native_videos']:
        jobs.append(dict(name=f'capture_video{video}',maximum_seconds=3600,log=str(a.out/f'video{video}.log'),
            command=[sys.executable,'-u','geo_tools/capture_cir_missing_native.py','--video',str(video),*common,
                '--stage-root',str(destination),'--out',str(destination/f'scene{video}')]))
    allocation=dict(stage='P5_MISSING12_FULL_NATIVE_INPUT_CAPTURE_V1',source_root=str(ROOT),source_commit=head,
        physical_GPUs=[6,7,8,9],maximum_GPU_wall_hours=6.,maximum_stage_wall_seconds=14400,jobs=jobs,
        estimate='Registered12 missing scenes only;30702 actual images;6GPUh finite bound,8GiB aggregate retained output,30GiB disk reserve.')
    path=a.out/'ALLOCATION.json';save_json(path,allocation)
    ledger=json.loads(a.root_ledger.read_text())
    if any(s['status'].startswith('RUNNING') for s in ledger['stages']):raise ValueError('Another writer appeared; do not race')
    if ledger['measured_GPU_wall_hours']+6>ledger['allocated_total_GPU_wall_hours']:
        before=ledger['allocated_total_GPU_wall_hours'];ledger['allocated_total_GPU_wall_hours']+=6
        ledger.setdefault('registered_additional_allocations',[]).append(dict(stage=allocation['stage'],new_GPU_wall_hours=6,
            previous_total=before,reason='Latest CIR task authorizes separately bounded complete stages without a total-hour ceiling; no old costs reset.',allocation_sha256=sha256(path)))
    save(a.root_ledger,ledger);state.update(status='RUNNING_FULL_MISSING12_NATIVE_CAPTURE',allocation_sha256=sha256(path));save_json(a.out/'STATE.json',state)
    with (a.out/'batch.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','geo_tools/run_cir_batch.py','--allocation',str(path),
            '--sha256',sha256(path),'--ledger',str(a.root_ledger)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state['batch_pid']=process.pid;save_json(a.out/'STATE.json',state);code=process.wait()
    manifests={}
    if code==0:
        for video in protocol['missing_native_videos']:
            path=destination/f'scene{video}/manifest.json';spec=json.loads(path.read_text())
            if spec['status']!='COMPLETE_MISSING_NATIVE_INPUTS_PARITY_PASS_PENDING_SUPERVISION':raise ValueError('Incomplete scene capture')
            manifests[str(video)]=dict(path=str(path),sha256=sha256(path),indexed_store_sha256=spec['indexed_store_manifest_sha256'])
    state.update(status='COMPLETE_NATIVE_INPUTS_PENDING_OFFLINE_ADMISSION' if code==0 else 'FAILED_FULL_CAPTURE_EVIDENCE_RETAINED',
        exit_code=code,manifests=manifests,full24_training_admitted=False);save_json(a.out/'STATE.json',state)


if __name__=='__main__':main()
