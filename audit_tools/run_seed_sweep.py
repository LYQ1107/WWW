#!/usr/bin/env python3
"""Run the predeclared A0 seed/jitter audit sequentially on GPU0."""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--jitter',action='store_true'); ap.add_argument('--seeds',nargs='+',type=int,default=[20260930,20261001,20261002]); args=ap.parse_args()
    root=args.root; env=os.environ.copy(); env.update({'GMT_AUDIT_TEST_JSON':str(root/'audit/cache/sanity_subset_test.json'),'GMT_AUDIT_TEST_IMAGE_ROOT':str(root/'datasets/VisionTrack/images/test'),'CUDA_VISIBLE_DEVICES':'0'})
    if not args.jitter: env['GMT_AUDIT_DISABLE_TEST_JITTER']='1'
    for seed in args.seeds:
        name=f'A0_seed_{seed}_'+('jitter' if args.jitter else 'nojitter')
        cmd=[str(root/'audit_tools/run_audit_inference.sh'),name,'SEED',str(seed),'INPUT.VIDEO.TEST_LEN','40','MODEL.ASSO_HEAD.WITH_BANK','True','MODEL.ASSO_HEAD.BANK_SIZE','10']
        subprocess.run(cmd,cwd=root,env=env,check=True)
if __name__=='__main__': main()
