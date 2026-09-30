#!/usr/bin/env python3
"""Run the repository MATLAB cross-view evaluator for an audit run.

Only the already generated run files and read-only VisionTrack GT are staged;
the evaluator and formulas are imported from MOTChallengeEvalKit_cv_test.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import collections
import collections.abc
from pathlib import Path
import numpy as np

def stage(run: Path, gt_source: Path, out: Path):
    if out.exists(): shutil.rmtree(out)
    inp_gt=out/'input/gt'; inp_tr=out/'input/track'; raw=run/'raw_predictions'
    for scene in sorted(p.name for p in raw.iterdir() if p.is_dir()):
        (inp_gt/scene/'gt').mkdir(parents=True,exist_ok=True); (inp_tr/scene).mkdir(parents=True,exist_ok=True)
        for pred in sorted((raw/scene).glob('View*.txt')):
            view=pred.name
            src=gt_source/scene/'gt'/view
            if not src.exists(): raise FileNotFoundError(src)
            shutil.copy2(src,inp_gt/scene/'gt'/view); shutil.copy2(pred,inp_tr/scene/view)
    prepared=out/'official_prepared'; prepared.mkdir(parents=True)
    # Equivalent to the released prepare_cross_view_eval_official.py, with
    # explicit paths and deterministic scene ordering.
    scenes=sorted(p.name for p in inp_gt.iterdir() if p.is_dir()); (prepared/'seqs.txt').write_text('MOT16\n'+'\n'.join(scenes)+'\n')
    for d in ('gt','track','gt_cvma','track_cvma'): (prepared/d).mkdir()
    for scene in scenes:
        gts=sorted((inp_gt/scene/'gt').glob('View*.txt')); nviews=len(gts); fr_cnt=0
        fg=open(prepared/'gt'/f'{scene}.txt','w'); ft=open(prepared/'track'/f'{scene}.txt','w'); fgc=open(prepared/'gt_cvma'/f'{scene}.txt','w'); ftc=open(prepared/'track_cvma'/f'{scene}.txt','w')
        for m,gp in enumerate(gts):
            gt=np.loadtxt(gp,delimiter=' ',dtype=float,ndmin=2); tr=np.loadtxt(inp_tr/scene/gp.name,delimiter=',',dtype=float,ndmin=2)
            max_fr=0
            for z in gt:
                fr,tid,x,y,w,h=z[:6]; max_fr=max(max_fr,int(fr)); fg.write(f'{int(fr)+fr_cnt}, {int(tid)}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, -1\n'); fgc.write(f'{int(fr)*nviews+m}, {int(tid)}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, -1\n')
            for z in tr:
                fr,tid,x1,y1,x2,y2=z[:6]; max_fr=max(max_fr,int(fr)); w=x2-x1; h=y2-y1; ft.write(f'{int(fr)+fr_cnt}, {int(tid)}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, -1\n'); ftc.write(f'{int(fr)*nviews+m}, {int(tid)}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, -1\n')
            fr_cnt += max_fr+1
        for f in (fg,ft,fgc,ftc): f.close()
    return prepared

def run_metric(root, prepared, kind, out):
    kit=root/'MOTChallengeEvalKit_cv_test'; sys.path.insert(0,str(kit/'MOT')); sys.path.insert(0,str(kit))
    # The bundled R2020-era evaluator imports Iterable from collections;
    # provide the Python 3.10 compatibility alias without changing formulas.
    collections.Iterable = collections.abc.Iterable
    from evalMOT import MOT_evaluator
    # MOT_metrics.py follows the release evaluator and adds
    # ``matlab_devkit/`` relative to the process cwd.  Run it from the kit
    # root so the bundled MATLAB functions are found by every worker.
    os.chdir(kit)
    obj=MOT_evaluator(); gt=prepared/('gt_cvma' if kind=='CVMA' else 'gt'); track=prepared/('track_cvma' if kind=='CVMA' else 'track'); save=out/f'{kind.lower()}_pkl'; save.mkdir(parents=True,exist_ok=True)
    overall, seqs=obj.run(benchmark_name='MOT16',gt_dir=str(gt),res_dir=str(track),seq_file=str(prepared/'seqs.txt'),save_pkl=str(save),eval_mode='test')
    # The kit stores percentages in attributes; preserve its complete row.
    row={k:v for k,v in overall.__dict__.items() if isinstance(v,(int,float,np.integer,np.floating))}
    (out/f'{kind.lower()}_summary.json').write_text(json.dumps(row,indent=2,default=float)+'\n'); return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--gt-source',type=Path,default=Path(__file__).resolve().parents[1]/'MOTChallengeEvalKit_cv_test/data/eval/vision/gt'); args=ap.parse_args(); root=Path(__file__).resolve().parents[1]; args.run=args.run.resolve(); args.output=args.output.resolve(); args.gt_source=args.gt_source.resolve(); args.output.mkdir(parents=True,exist_ok=True); prep=stage(args.run,args.gt_source,args.output); rows={}
    for kind in ('CVIDF1','CVMA'): rows[kind]=run_metric(root,prep,kind,args.output)
    (args.output/'manifest.json').write_text(json.dumps({'run':str(args.run),'prepared':str(prep),'metrics':rows,'matlab_env':'source /data3/liuyeqiang/matlab_runtime_setup/use_matlab_r2020a.sh'},indent=2,default=float)+'\n')
if __name__=='__main__': main()
