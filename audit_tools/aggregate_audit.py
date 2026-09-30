#!/usr/bin/env python3
"""Validate and index the completed challenge-audit artifacts.

The evidence matrix is written only after the predeclared runs finish.  This
helper refuses to replace a completed matrix with a placeholder.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args(); r=args.root
    matrix=r/'audit/results/CHALLENGE_EVIDENCE_MATRIX.csv'; matrix_md=r/'audit/CHALLENGE_EVIDENCE_MATRIX.md'
    if not matrix.exists() or not matrix_md.exists(): raise FileNotFoundError('complete evidence matrix is not present')
    evidence=pd.read_csv(matrix)
    required={'Hypothesis','Evidence','Effect size','Cross-scene consistency','Supported?'}
    missing=required.difference(evidence.columns)
    if missing: raise ValueError(f'evidence matrix missing columns: {sorted(missing)}')
    paths={'a0':'A0_seed_variance.csv','a1':'A1_quality_stratification.csv','a2':'A2_history_sweep.csv','a3':'A3_history_sweep.csv','a4':'A4_error_survival.csv','a5':'A5_entity_retrieval.csv'}
    rows={key:int(pd.read_csv(r/'audit/results'/name).shape[0]) for key,name in paths.items() if (r/'audit/results'/name).exists()}
    manifest={'matrix_rows':int(len(evidence)),'matrix':str(matrix),'final_report':str(r/'audit/FINAL_CHALLENGE_AUDIT.md'),'artifact_rows':rows,'status':'complete'}
    (r/'audit/manifests/aggregate_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
