#!/usr/bin/env python3
"""Source-bound CPU preflight receipt, including exact command and JUnit SHA."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from worldobject_v4_audit_common import code_fingerprint,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];a.runtime.mkdir(parents=True,exist_ok=True)
    fp=code_fingerprint(repo);name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=a.runtime/'cpu_checks'/name;out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','pytest','worldmot/tests','-q','--disable-warnings','--junitxml='+str(out/'junit.xml')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(repo/'worldmot')};started=time.perf_counter()
    with (out/'pytest.log').open('x') as log:result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    from worldmot.data.worldobject_cache import sha256
    unchanged=code_fingerprint(repo)==fp
    receipt={'status':'PASSED' if result.returncode==0 and unchanged else 'FAILED','source_sha256':fp,
        'source_unchanged_during_tests':unchanged,'exit_code':result.returncode,'command':command,
        'junit':str(out/'junit.xml'),'junit_sha256':sha256(out/'junit.xml') if (out/'junit.xml').exists() else None,
        'wall_s':time.perf_counter()-started,'real_GT_or_pretrained_weights_used':False,
        'synthetic_target_labels_used_in_unit_tests':True}
    write_json(a.runtime/'cpu_receipt.json',receipt);print(receipt)
    if receipt['status']!='PASSED':raise SystemExit(1)


if __name__=='__main__':main()
