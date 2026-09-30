#!/usr/bin/env python3
"""Collect TrackEval metrics and immutable run metadata into an audit CSV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True)
    args=ap.parse_args(); rows=[]
    for d in sorted(args.runs.iterdir()):
        if not d.is_dir(): continue
        p=d/'metrics.json'; m=d/'run_manifest.json'
        if not p.exists(): continue
        row=json.loads(p.read_text()); row['run']=d.name
        if m.exists(): row.update({f'run_{k}':v for k,v in json.loads(m.read_text()).items() if k not in row})
        rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,index=False)
    args.manifest.parent.mkdir(parents=True,exist_ok=True); args.manifest.write_text(json.dumps({'runs':len(rows),'output':str(args.output),'rows':rows},indent=2)+'\n')
    print(json.dumps({'runs':len(rows),'output':str(args.output)},indent=2))
if __name__=='__main__': main()
