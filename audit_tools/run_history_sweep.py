#!/usr/bin/env python3
"""Run the fixed A2/A3 audit matrix; no values are selected from GT."""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

def run(root, env, name, opts):
    subprocess.run([str(root/'audit_tools/run_audit_inference.sh'),name,*opts],cwd=root,env=env,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--matrix',choices=['a2','a3','all'],default='all'); ap.add_argument('--subset',action='store_true'); args=ap.parse_args(); root=args.root
    test_json = root/'audit/cache/sanity_subset_test.json' if args.subset else root/'datasets/VisionTrack/annotations/test.json'
    env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0','GMT_AUDIT_TEST_JSON':str(test_json),'GMT_AUDIT_TEST_IMAGE_ROOT':str(root/'datasets/VisionTrack/images/test')})
    # A2's five conditions are fixed before looking at outcomes.
    if args.matrix in ('a2','all'):
        for name, extra in [('A2_uniform_bankoff',[]),('A2_score_top75',['GMT_AUDIT_HISTORY_POLICY=score','GMT_AUDIT_HISTORY_KEEP_RATIO=0.75']),('A2_score_top50',['GMT_AUDIT_HISTORY_POLICY=score','GMT_AUDIT_HISTORY_KEEP_RATIO=0.50']),('A2_area_top75',['GMT_AUDIT_HISTORY_POLICY=area','GMT_AUDIT_HISTORY_KEEP_RATIO=0.75']),('A2_area_top50',['GMT_AUDIT_HISTORY_POLICY=area','GMT_AUDIT_HISTORY_KEEP_RATIO=0.50'])]:
            e=env.copy(); e['GMT_AUDIT_HISTORY_POLICY']=''; e['GMT_AUDIT_HISTORY_KEEP_RATIO']='1';
            for kv in extra:
                k,v=kv.split('=',1); e[k]=v
            run(root,e,name,['SEED','20260930','INPUT.VIDEO.TEST_LEN','40','MODEL.ASSO_HEAD.WITH_BANK','False','MODEL.ASSO_HEAD.BANK_SIZE','10'])
    if args.matrix in ('a3','all'):
        for n in (8,16,24,40): run(root,env,f'A3_len{n}_bankon',['SEED','20260930','INPUT.VIDEO.TEST_LEN',str(n),'MODEL.ASSO_HEAD.WITH_BANK','True','MODEL.ASSO_HEAD.BANK_SIZE','10'])
        run(root,env,'A3_len40_bankoff',['SEED','20260930','INPUT.VIDEO.TEST_LEN','40','MODEL.ASSO_HEAD.WITH_BANK','False','MODEL.ASSO_HEAD.BANK_SIZE','10'])
if __name__=='__main__': main()
