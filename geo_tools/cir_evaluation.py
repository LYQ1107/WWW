"""Versioned formatting only; unmodified TrackEval and official MATLAB formulas.

GT is loaded in this evaluation process only, after native predictions exist.
Never subclass away the official duplicate-identity validity check.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import ROOT,sha256,save_json
sys.path.insert(0,str(ROOT/'TrackEval'))
import trackeval

MATLAB=Path('/data1/liuyeqiang/MATLAB/R2020a/bin/matlab')
KIT=ROOT/'MOTChallengeEvalKit_cv_test/matlab_devkit'


def read_mot(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():result.append([float(v) for v in line.split(',')])
    return result


def validate_rows(rows,name):
    keys=Counter((int(r[0]),int(r[1])) for r in rows)
    if any(n>1 for n in keys.values()):raise ValueError('Unmodified metric input has duplicate frame/identity: '+name)
    if any(len(r)<6 or not np.isfinite(r[:6]).all() or r[0]<1 or r[4]<0 or r[5]<0 for r in rows):
        raise ValueError('Invalid MOT box/frame '+name)


def write_mot(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        for r in sorted(rows,key=lambda r:(r[0],r[1])):
            # Original official converter rounds XYWH to 2 decimals.
            tail=list(r[6:10])+[-1]*max(0,10-len(r))
            stream.write(f'{int(r[0])},{int(r[1])},'+','.join(f'{v:.2f}' for v in r[2:6])+','+','.join(str(v) for v in tail)+'\n')


def evaluate_trackeval(gt,trackers,lengths,base):
    for name,rows in gt.items():
        validate_rows(rows,name+' GT');validate_rows(trackers[name],name+' prediction')
        write_mot(base/'gt'/name/'gt/gt.txt',rows)
        write_mot(base/'trackers/GMT/data'/f'{name}.txt',trackers[name])
    ec=trackeval.Evaluator.get_default_eval_config()
    ec.update(USE_PARALLEL=False,PRINT_CONFIG=False,PRINT_RESULTS=False,PLOT_CURVES=False,OUTPUT_FOLDER=str(base/'output'))
    dc=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(GT_FOLDER=str(base/'gt'),TRACKERS_FOLDER=str(base/'trackers'),TRACKERS_TO_EVAL=['GMT'],
              BENCHMARK='VisionTrack',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,SEQ_INFO=lengths,
              PRINT_CONFIG=False,OUTPUT_FOLDER=str(base/'output'))
    dataset=trackeval.datasets.MotChallenge2DBox(dc)
    results,messages=trackeval.Evaluator(ec).evaluate([dataset],
        [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()])
    result=results[dataset.get_name()]['GMT']
    if result is None:raise RuntimeError(str(messages))
    def extract(c):
        return dict(HOTA=float(np.mean(c['HOTA']['HOTA'])*100),AssA=float(np.mean(c['HOTA']['AssA'])*100),
            DetA=float(np.mean(c['HOTA']['DetA'])*100),IDF1=float(c['Identity']['IDF1']*100),
            MOTA=float(c['CLEAR']['MOTA']*100),IDSW=int(c['CLEAR']['IDSW']),Frag=int(c['CLEAR']['Frag']))
    return dict(combined=extract(result['COMBINED_SEQ']['pedestrian']),
                per_sequence={k:extract(v['pedestrian']) for k,v in result.items() if k!='COMBINED_SEQ'},
                formula_sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'TrackEval/trackeval').rglob('*.py'))})


def flatten_official(scenes,gt,trackers,base):
    jobs=[]
    for scene,names in scenes.items():
        names=sorted(names)
        for convention in ('sequential','interleaved'):
            parts={'gt':[],'prediction':[]};offset=0
            for camera,name in enumerate(names):
                length=max([int(r[0]) for r in gt[name]+trackers[name]]+[0])+1
                for rows,key in ((gt[name],'gt'),(trackers[name],'prediction')):
                    for r in rows:
                        frame=int(r[0])+offset if convention=='sequential' else int(r[0])*len(names)+camera
                        parts[key].append([frame,*r[1:6],-1,-1,-1,-1])
                offset+=length
            folder=base/convention;paths={}
            for key,rows in parts.items():
                paths[key]=folder/f'{scene}_{key}.txt';write_mot(paths[key],rows)
            gt_directory=paths['gt'].with_suffix('');gt_directory.mkdir()
            total=max([int(r[0]) for rows in parts.values() for r in rows]+[1])
            (gt_directory/'seqinfo.ini').write_text('[Sequence]\nname='+scene+'\nimDir=img1\nframeRate=30\nseqLength='+str(total)+'\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n')
            jobs.append(dict(scene=scene,convention=convention,groundtruth=str(paths['gt'].resolve()),
                prediction=str(paths['prediction'].resolve()),gt_directory=str(gt_directory.resolve())))
    return jobs


def evaluate_matlab(scenes,gt,trackers,base):
    if not MATLAB.is_file() or not list(KIT.rglob('*.mexa64')):raise RuntimeError('Native MATLAB/MEX unavailable')
    for name in gt:validate_rows(gt[name],name+' GT');validate_rows(trackers[name],name+' prediction')
    jobs=flatten_official(scenes,gt,trackers,base)
    config=base/'CONFIG.json';native=base/'NATIVE.json';log=base/'matlab.log'
    save_json(config,dict(kit=str(KIT),jobs=jobs,output=str(native.resolve()),benchmark='VisionTrack'))
    script=ROOT/'geo_tools/cir_official_metrics.m'
    before={str(p.relative_to(KIT)):sha256(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.suffix in ('.m','.cpp','.mexa64')}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CIR_MATLAB_CONFIG=str(config.resolve()),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('x') as stream:
        subprocess.run([str(MATLAB),'-batch',"run('"+str(script)+"')"],stdout=stream,stderr=subprocess.STDOUT,
                       env=env,check=True,timeout=600)
    values=json.loads(native.read_text())
    after={str(p.relative_to(KIT)):sha256(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.suffix in ('.m','.cpp','.mexa64')}
    if before!=after:raise RuntimeError('Official evaluator mutated')
    def aggregate(selected):
        ids=[r['Identity'] for r in selected if r['convention']=='sequential']
        clear=[r['CLEAR'] for r in selected if r['convention']=='interleaved']
        counts={k:sum(v[k] for v in ids) for k in ('IDTP','IDFP','IDFN','n_gt','n_tr')}
        errors={k:sum(v[k] for v in clear) for k in ('fn','fp','id_switches','tp')}
        return dict(CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']) if counts['n_gt']+counts['n_tr'] else None,
                    CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']) if counts['n_gt'] else None,
                    sequential_identity_counts=counts,interleaved_CLEAR_counts=errors)
    return dict(combined=aggregate(values['results']),per_scene={s:aggregate([r for r in values['results'] if r['scene']==s]) for s in scenes},
                native_report=dict(path=str(native),sha256=sha256(native)),MATLAB_version=values['version'],
                native_function=values['native_official_function'],toolkit_sha256=before,adapter_sha256=sha256(script),
                convention='official converter max(GT,pred)+1 sequential blocks and frame*nviews+camera interleaved; XYWH two decimals; benchmark VisionTrack; all formulas unchanged')
