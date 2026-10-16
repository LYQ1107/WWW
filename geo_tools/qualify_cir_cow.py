"""Bounded real-image COW qualification, queued after the active CIR writer.

Engineering only. Never changes a live allocation or admits a tracking method.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, save_json, sha256


def probes(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol=json.loads((ROOT/'results/CIR_GMT/P4_NATIVE_INTEGRATION_PROTOCOL.json').read_text())
    training=json.loads((ROOT/'results/CIR_GMT/TRAINING_RESULTS.json').read_text())
    commands=[]
    for backend in ('full','cow_engineering'):
        commands.append((f'guard_fault_{backend}', ['geo_tools/verify_cir_guard_rollback.py',
            '--commit-snapshot',backend]))
        commands.append((f'guard_prefix_{backend}', ['geo_tools/verify_cir_official_parity.py',
            '--policy','guard','--commit-snapshot',backend]))
    for mode in ('risk_mlp','risk_grouped'):
        spec=protocol['learned_models_frozen'][mode]
        checkpoint=training['models'][spec['architecture']]['checkpoint']
        if checkpoint['sha256']!=spec['checkpoint_sha256'] or sha256(checkpoint['path'])!=spec['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint SHA mismatch')
        commands.append((mode+'_fault_cow', ['geo_tools/verify_cir_guard_rollback.py',
            '--policy',mode,'--commit-snapshot','cow_engineering',
            '--risk-checkpoint',checkpoint['path'],
            '--risk-checkpoint-sha256',spec['checkpoint_sha256']]))
    commands.append(('off_prefix',['geo_tools/verify_cir_official_parity.py']))
    evidence={}
    for name, command in commands:
        command=[sys.executable,*command,'--groups','40','--out',str(out/name)]
        with (out/(name+'.log')).open('x') as log:
            run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if run.returncode:
            save_json(out/'RESULT.json',dict(status='FAIL_NO_ADMISSION',failed=name,exit_code=run.returncode))
            raise RuntimeError('Actual native engineering qualification failed: '+name)
        result=out/name/'RESULT.json'
        evidence[name]=dict(path=str(result),sha256=sha256(result),status=json.loads(result.read_text())['status'])
    checks={}
    for arm in ('RAW_PASSIVE','GUARD_FAULT_RECOVERY'):
        full=json.loads((out/'guard_fault_full'/(arm+'.json')).read_text())
        cow=json.loads((out/'guard_fault_cow_engineering'/(arm+'.json')).read_text())
        for field in ('predictions','states','rng','costs'):
            checks[arm+'_'+field]=full[field]==cow[field]
    for arm in ('CIR_GUARD_OFFICIAL','CIR_GUARD_CAUSAL'):
        for name in ('raw_predictions.json','stock_predictions.json'):
            checks[arm+'_'+name]=json.loads((out/'guard_prefix_full'/arm/name).read_text())==json.loads((out/'guard_prefix_cow_engineering'/arm/name).read_text())
    report=dict(status='PASS_REAL_PREFIX_COW_ENGINEERING' if all(checks.values()) else 'FAIL_NO_ADMISSION',
        checks=checks,evidence=evidence,GT_in_inference=False,scientific_success=False,
        full_video_qualified=False,formal_method_admitted=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save_json(out/'RESULT.json',report)
    if not all(checks.values()):raise RuntimeError('Full/COW actual state or future mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ledger',type=Path);p.add_argument('--probe-only',action='store_true')
    p.add_argument('--gpu',type=int,default=8);a=p.parse_args()
    if a.probe_only:
        probes(a.out);return
    if not a.ledger or a.out.exists():raise ValueError('Fresh output and existing CIR ledger required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Freeze a clean source first')
    queued=a.out.with_name(a.out.name+'_queue');queued.mkdir(exist_ok=False)
    start=time.monotonic()
    while True:
        ledger=json.loads(a.ledger.read_text())
        running=[s for s in ledger['stages'] if s['status'].startswith('RUNNING')]
        for stage in running:
            if not Path('/proc/'+str(stage['supervisor_pid'])).exists():
                raise RuntimeError('Live-ledger supervisor missing; no restart or cost overwrite')
        used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        save_json(queued/'STATE.json',dict(status='WAIT_VERIFIED_ACTIVE_STAGE_OR_GPU' if running or used else 'LAUNCHING',
            pid=os.getpid(),waiting_seconds=time.monotonic()-start,active=[s['stage'] for s in running],source_commit=head))
        if not running and not used:break
        if time.monotonic()-start>21660:raise TimeoutError('Engineering qualification queue expired; no restart')
        time.sleep(15)
    command=[sys.executable,'geo_tools/run_cir_guarded.py','--ledger',str(a.ledger),
        '--stage','P2_COW_REAL_NATIVE_ENGINEERING_V1','--gpu',str(a.gpu),'--seconds','3600',
        '--log',str(queued/'probe.log'),'--cwd',str(ROOT),'--source-commit',head,
        '--',sys.executable,'geo_tools/qualify_cir_cow.py','--probe-only','--out',str(a.out)]
    code=subprocess.call(command,cwd=ROOT)
    save_json(queued/'STATE.json',dict(status='COMPLETE' if code==0 else 'FAILED_EVIDENCE_RETAINED',
        pid=os.getpid(),exit_code=code,source_commit=head,scientific_success=False))
    raise SystemExit(code)


if __name__=='__main__':main()
