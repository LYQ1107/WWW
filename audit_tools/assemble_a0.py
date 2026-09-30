#!/usr/bin/env python3
"""Assemble A0 seed and jitter-ablation tables after run evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def collect(root, names):
    rows=[]
    for name in names:
        d=root/'audit/runs'/name; m=d/'run_manifest.json'; e=d/'evaluation/metrics.json'
        if not e.exists(): continue
        x=json.loads(e.read_text()); x['seed']=int(name.split('_')[2]); x['condition']='no-jitter' if 'nojitter' in name else 'official-jitter'; x['run']=name
        if m.exists(): x['runtime']=json.loads(m.read_text()).get('runtime_sec')
        rows.append(x)
    return rows
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args(); root=args.root
    names=[p.name for p in sorted((root/'audit/runs').glob('A0_seed_*')) if p.is_dir()]
    rows=collect(root,names); df=pd.DataFrame(rows)
    out=root/'audit/results/A0_seed_variance.csv'; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    ab=df[df.condition.isin(['official-jitter','no-jitter'])].copy(); ab.to_csv(root/'audit/results/A0_jitter_ablation.csv',index=False)
    (root/'audit/manifests/A0_seed_sweep.json').write_text(json.dumps({'runs':rows,'selection':'three automatic annotation-count percentile scenes','test_len':40,'bank':True,'bank_size':10},indent=2)+'\n')
if __name__=='__main__': main()
