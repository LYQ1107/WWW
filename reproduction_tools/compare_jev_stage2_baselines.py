"""Real pooled TrackEval over all six cameras; no mean-of-video shortcut."""
import argparse,subprocess,shutil
import numpy as np
from jev_phase13_learning import *
from run_jev_phase10_closed_loop import metrics

def pooled(variant,seed,phase='formal',no_calibration=False):
    name=f'{variant}_seed{seed}'+('_no_calibration' if no_calibration else '');root=OUT/f'{phase}_online_v1'/name;out=OUT/f'{phase}_pooled_v1'/name;result=out/'RESULT.json'
    if result.exists():return json.loads(result.read_text())
    records=[json.loads((root/f'video{v:02d}'/'RESULT.json').read_text()) for v in VAL];assert all(r['status']=='COMPLETE' for r in records);out.mkdir(parents=True,exist_ok=True);values={}
    for field,target in [('raw_predictions','RAW_PREDICTIONS.json'),('canonical_predictions','CANONICAL_PREDICTIONS.json')]:
        rows=[]
        for r in records:p=r[field];assert sha(p['path'])==p['SHA256'];rows.extend(json.loads(Path(p['path']).read_text()))
        save(out/target,rows);values[field],evaluation=metrics(out/target,VAL,out/field);values[field+'_evaluator']={'path':str(evaluation/'metrics.json'),'SHA256':sha(evaluation/'metrics.json')}
    environment_audit=REPORTS/'MATLAB_ENVIRONMENT_AUDIT.json'
    official=({**json.loads(environment_audit.read_text())['official_MATLAB_evaluation'],'environment_audit':{'path':str(environment_audit),'SHA256':sha(environment_audit)}} if environment_audit.exists() else {'status':'NOT_RUN','metrics':None,'reason':'native runtime availability not yet inspected; do not relabel Python TrackEval as official MATLAB scores'})
    crossview={'official_MATLAB':official,'TrackEval_flattened_adaptation':None}
    # This is a separately labelled repository adaptation of cross-camera
    # conventions, never presented as a run of the official MATLAB package.
    prepared=out/'raw_predictions/tracking_eval_runtime_state/native/prepared/manifest.json'
    cvout=out/'crossview_adapted'
    subprocess.run([PYTHON,str(ROOT/'reproduction_tools/evaluate_crossview_visiontrack.py'),'--manifest',str(prepared),'--output',str(cvout),'--allow-duplicate-gt'],cwd=ROOT,check=True)
    candidates=list(cvout.glob('*.json'));crossview['TrackEval_flattened_adaptation']={'reports':[{'path':str(p),'SHA256':sha(p),'values':json.loads(p.read_text())} for p in candidates],'scope':'actual repository TrackEval sequential-camera CVIDF1/interleaved CVMA adaptation, raw TRAIN GT permissive duplicate policy; not official native MATLAB'}
    r={'status':'COMPLETE','binding':binding(),'variant':variant,'seed':seed,'phase':phase,'strict_pooled_TrackEval':values['raw_predictions'],'canonical_pooled_TrackEval':values['canonical_predictions'],'strict_evaluator':values['raw_predictions_evaluator'],'canonical_evaluator':values['canonical_predictions_evaluator'],'video_results':records,'all_six_cameras_pooled':True,'per_video_mean_used_as_pooled':False,'crossview':crossview,'heldout':'SEALED'};save(result,r);return r

def main(phase,only):
    protect();cases=[('cosine',20261009)] if only=='cosine' else [(v,s) for v in (VARIANTS if phase=='formal' else ['full','motip','camel','set_transformer']) for s in SEEDS];results=[];pending=[]
    for v,s in cases:
        if all((OUT/f'{phase}_online_v1'/f'{v}_seed{s}'/f'video{x:02d}'/'RESULT.json').exists() for x in VAL):results.append(pooled(v,s,phase))
        else:pending.append({'variant':v,'seed':s,'status':'NOT_RUN','reason':'all three full-video native outcomes not yet complete','metrics':None})
    summary={}
    for v in sorted({r['variant'] for r in results}):
        rr=[r for r in results if r['variant']==v];summary[v]={key:{'mean':float(np.mean([r['strict_pooled_TrackEval'][key] for r in rr])),'std_seed':float(np.std([r['strict_pooled_TrackEval'][key] for r in rr])),'seeds':[r['seed'] for r in rr]} for key in ['HOTA','AssA','IDF1','MOTA','IDSW','Frag']}
    report={'status':'COMPLETE' if not pending else 'IN_PROGRESS','binding':binding(),'phase':phase,'results':results,'pending':pending,'strict_seed_summary':summary,'GMT_Original_separate_full_system':json.loads((REPORTS/'GMT_BASELINE_FREEZE.json').read_text()),'comparison_scope':'all learned methods fresh Stage1/VFCE1024, equal data/legal solver, 20000 updates or separately24000 on-policy total; original GMT has Stage2-trained perception/RPCE and is not same-frontend ablation','development_preexposed':True,'heldout':'SEALED'}
    name='B1_COSINE_COMPARISON' if only=='cosine' else 'BASELINE_COMPARISON' if phase=='formal' else 'ONPOLICY_COMPARISON';save(REPORTS/(name+'.json'),report);print('PHASE13_POOLED_COMPARISON',name,summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['formal','onpolicy'],default='formal');p.add_argument('--only',choices=['all','cosine'],default='all');a=p.parse_args();main(a.phase,a.only)
