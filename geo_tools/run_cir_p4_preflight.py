"""Finite real-image P4 checks before any learned full-video comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,CHECKPOINT,save_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--original-full',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Clean frozen qualification source required')
    a.out.mkdir(parents=True);fit=json.loads(a.training_report.read_text());state=dict(status='RUNNING',completed=[],GT_read=False)
    def run(name,args):
        state['current']=name;save_json(a.out/'RESULT.json',state)
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True);state['completed'].append(name)
    try:
        run('off_prefix',['geo_tools/verify_cir_off_parity.py','--out',str(a.out/'off_prefix')])
        run('reference_prefix',['geo_tools/verify_cir_official_parity.py','--policy','guard_reference','--out',str(a.out/'guard_reference_prefix')])
        for mode,architecture in (('risk_mlp','ordinary_mlp'),('risk_grouped','grouped_risk')):
            cp=fit['models'][architecture]['checkpoint'];extra=['--policy',mode,'--risk-checkpoint',cp['path'],'--risk-checkpoint-sha256',cp['sha256']]
            run(mode+'_prefix',['geo_tools/verify_cir_official_parity.py',*extra,'--out',str(a.out/(mode+'_prefix'))])
            run(mode+'_fault',['geo_tools/verify_cir_guard_rollback.py',*extra,'--out',str(a.out/(mode+'_fault'))])
        common=['geo_tools/prepare_cir_p4_online.py','--off-prefix',str(a.out/'off_prefix/RESULT.json'),
            '--training-report',str(a.training_report),'--original-full',str(a.original_full)]
        for mode in ('guard_reference','risk_mlp','risk_grouped'):common+=['--policy-prefix',str(a.out/(mode+'_prefix/RESULT.json'))]
        for mode in ('risk_mlp','risk_grouped'):common+=['--fault-proof',str(a.out/(mode+'_fault/RESULT.json'))]
        run('engineering_off_gate',common+['--out',str(a.out/'engineering_off_gate.json')])
        save_json(a.out/'off04_spec.json',dict(split='train',method='off',video_ids=[4],qualification=str(a.out/'engineering_off_gate.json'),out_root=str(a.out/'official_off04')))
        run('complete_off04',['test_net.py','--config-file','configs/VISION_test.yaml','--num-gpus','1','--cir-run-spec',str(a.out/'off04_spec.json'),
            'MODEL.WEIGHTS',str(CHECKPOINT),'OUTPUT_DIR',str(a.out/'setup_off04')])
        run('full_dev_gate',common+['--off-full',str(a.out/'official_off04/video4/manifest.json'),'--out',str(a.out/'full_online_gate.json')])
        state.update(status='PASS_P4_NATIVE_PREFIXES_FROZEN_HEAD_FAULTS_AND_COMPLETE_OFF',gate_sha256=sha256(a.out/'full_online_gate.json'))
    except BaseException as error:state.update(status='FAILED_NO_ADMISSION',error=repr(error));raise
    finally:save_json(a.out/'RESULT.json',state)


if __name__=='__main__':main()
