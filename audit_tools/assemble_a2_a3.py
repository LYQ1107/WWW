#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def row(root,name):
 d=root/'audit/runs'/name; p=d/'evaluation/metrics.json'; m=d/'run_manifest.json'
 if not p.exists(): return None
 x=json.loads(p.read_text()); x['run']=name; x['scope']='sanity_subset_3_scenes'
 x['seed']=20260930; x['inference_time_sec']=json.loads(m.read_text()).get('runtime_sec') if m.exists() else None
 cv=d/'cv_official/manifest.json'
 if cv.exists():
  z=json.loads(cv.read_text()).get('metrics',{})
  x['CVIDF1_if_run']=z.get('CVIDF1',{}).get('IDF1')
  x['CVMA_if_run']=z.get('CVMA',{}).get('IDF1')
 return x
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args(); r=args.root
 a2=[]
 for n,pol,k in [('A2_uniform_bankoff','uniform',1.0),('A2_score_top75','score',.75),('A2_score_top50','score',.5),('A2_area_top75','area',.75),('A2_area_top50','area',.5)]:
  x=row(r,n)
  if x: x.update({'policy':pol,'keep_ratio':k,'bank_enabled':False,'test_len':40}); x.setdefault('CVIDF1_if_run',None); x.setdefault('CVMA_if_run',None); a2.append(x)
 a3=[]
 for n,l,b in [('A3_len8_bankon',8,True),('A3_len16_bankon',16,True),('A3_len24_bankon',24,True),('A3_len40_bankon',40,True),('A3_len40_bankoff',40,False)]:
  x=row(r,n)
  if x: x.update({'test_len':l,'bank_enabled':b,'bank_size':10 if b else 0}); x.setdefault('CVIDF1_if_run',None); x.setdefault('CVMA_if_run',None); a3.append(x)
 for name,rows in [('A2_history_sweep.csv',a2),('A3_history_sweep.csv',a3)]:
  out=r/'audit/results'/name; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
 (r/'audit/manifests/A2_A3_sweeps.json').write_text(json.dumps({'a2':a2,'a3':a3,'scope':'automatically selected 3-scene subset','selection_fixed_before_outcomes':True},indent=2)+'\n')
if __name__=='__main__': main()
