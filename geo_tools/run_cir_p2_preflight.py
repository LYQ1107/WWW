"""Finite source-frozen checks before the registered full12 online matrix.

Run under the owned GPU supervisor. No GT or development metric is read.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,CHECKPOINT,save_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--original-full',type=Path,required=True)
    p.add_argument('--guard-prefix',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Clean frozen preflight source required')
    a.out.mkdir(parents=True);begun=time.monotonic();completed=[]
    report=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        GT_read=False,metrics_read=False,completed=completed)
    def run(name,command):
        report.update(current=name,elapsed_seconds=time.monotonic()-begun);save_json(a.out/'RESULT.json',report)
        print(json.dumps(dict(stage=name,status='STARTING')),flush=True)
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,check=True)
        completed.append(name);report.update(completed=completed,elapsed_seconds=time.monotonic()-begun)
        save_json(a.out/'RESULT.json',report)
    try:
        run('off_keep_rollback_prefix',['geo_tools/verify_cir_off_parity.py','--out',str(a.out/'off_prefix')])
        for mode in ('margin','threshold','bias2'):
            run(mode+'_shared_core_prefix',['geo_tools/verify_cir_official_parity.py','--policy',mode,'--out',str(a.out/(mode+'_prefix'))])
        run('guarded_commit_rollback',['geo_tools/verify_cir_guard_rollback.py','--out',str(a.out/'guard_rollback')])
        common=['geo_tools/prepare_cir_p2_online.py','--off-prefix',str(a.out/'off_prefix/RESULT.json'),
            '--guard-prefix',str(a.guard_prefix),'--original-full',str(a.original_full)]
        run('engineering_off_gate',common+['--out',str(a.out/'engineering_off_gate.json')])
        spec=dict(split='train',method='off',video_ids=[4],qualification=str(a.out/'engineering_off_gate.json'),out_root=str(a.out/'official_off04'))
        save_json(a.out/'off04_spec.json',spec)
        run('complete_off04',['test_net.py','--config-file','configs/VISION_test.yaml','--num-gpus','1',
            '--cir-run-spec',str(a.out/'off04_spec.json'),'MODEL.WEIGHTS',str(CHECKPOINT),'OUTPUT_DIR',str(a.out/'setup_off04')])
        final=common+['--off-full',str(a.out/'official_off04/video4/manifest.json'),
            '--rollback',str(a.out/'guard_rollback/RESULT.json'),'--out',str(a.out/'full_online_gate.json')]
        for mode in ('margin','threshold','bias2'):final.extend(['--control-prefix',str(a.out/(mode+'_prefix/RESULT.json'))])
        run('registered_full12_gate',final)
        report.update(status='PASS_ALL_PREFIXES_ROLLBACK_COMPLETE_OFF_AND_FULL12_GATE',
            gate=dict(path=str(a.out/'full_online_gate.json'),sha256=sha256(a.out/'full_online_gate.json')))
    except BaseException as error:
        report.update(status='FAILED_NO_ADMISSION',error=repr(error));raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-begun);save_json(a.out/'RESULT.json',report)


if __name__=='__main__':main()
