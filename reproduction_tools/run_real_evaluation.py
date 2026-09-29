"""Run both original evaluators once on verified trained-model predictions."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,json,math,os,shlex,subprocess,time
R=Path(__file__).resolve().parents[1];repo=R/'code/GMT'
lock=(R/'manifests/real_evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prepared=json.loads((R/'manifests/evaluation_inputs_complete.json').read_text());assert prepared['status']=='PASS'
root=Path(prepared['output']);output=R/'outputs/real_evaluation'
if output.exists():raise RuntimeError('Refusing to overwrite an existing real evaluation')
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in prepared['records']:
 assert h(row['raw_path'])==row['raw_sha256']
 assert h(root/'trackeval/gt'/row['sequence']/'gt/gt.txt')==row['gt_evaluation_sha256']
 assert h(root/'trackeval/trackers/GMT/data'/(row['sequence']+'.txt'))==row['converted_sha256']
output.mkdir();jobs=[]
def run(label,args):
 log=R/'logs'/(label+'.log');start=time.monotonic()
 with (R/'reports/commands_used.sh').open('a') as f:f.write('\n# '+label+'\n'+shlex.join(args)+'\n')
 with log.open('xb') as f:
  child=subprocess.Popen(args,cwd=repo,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  job={'label':label,'pid':child.pid,'args':args,'log':str(log),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};jobs.append(job)
  (R/'manifests/real_evaluation_state.json').write_text(json.dumps({'current':label,'jobs':jobs},indent=2));rc=child.wait()
 job.update(exit_code=rc,duration_seconds=time.monotonic()-start)
 (R/'manifests/real_evaluation_state.json').write_text(json.dumps({'current':label+'_exited','jobs':jobs},indent=2))
 if rc:raise RuntimeError(label+' failed; inspect '+str(log))
run('real_trackeval',[str(R/'tools/miniconda3/envs/GMT/bin/python'),str(repo/'TrackEval/scripts/run_mot_challenge.py'),'--GT_FOLDER',str(root/'trackeval/gt'),'--TRACKERS_FOLDER',str(root/'trackeval/trackers'),'--TRACKERS_TO_EVAL','GMT','--BENCHMARK','vision','--SPLIT_TO_EVAL','test','--SKIP_SPLIT_FOL','True','--METRICS','HOTA','CLEAR','Identity','--USE_PARALLEL','False'])
for label,gt,track in [('cvidf1','gt','track'),('cvma','gt_cvma','track_cvma')]:
 run('real_'+label,[str(R/'tools/miniconda3/envs/GMT_eval/bin/python'),str(R/'tools/run_official_cross_eval.py'),'--gt-dir',str(root/'crossview'/gt),'--res-dir',str(root/'crossview'/track),'--seq-file',str(root/'crossview/seqs.txt'),'--output',str(output/label)])
detailed=root/'trackeval/trackers/GMT/pedestrian_detailed.csv';rows=list(csv.DictReader(detailed.open()))
expected={x['sequence'] for x in prepared['records']};assert {x['seq'] for x in rows}==expected|{'COMBINED'}
combined=[x for x in rows if x['seq']=='COMBINED'];assert len(combined)==1;combined=combined[0]
scenes={x.rsplit('_',1)[0] for x in expected};cross={}
for label in ['cvidf1','cvma']:
 cross[label]=json.loads((output/label/'metrics.json').read_text());assert set(cross[label]['sequences'])==scenes
values={metric:float(combined[column])*100 for metric,column in {'HOTA':'HOTA___AUC','IDF1':'IDF1','AssA':'AssA___AUC','MOTA':'MOTA'}.items()}
values.update(CVIDF1=float(cross['cvidf1']['metrics']['IDF1']),CVMA=float(cross['cvma']['metrics']['MOTA']))
assert all(math.isfinite(v) for v in values.values())
targets={'HOTA':66.2,'IDF1':82.1,'AssA':69.4,'CVMA':75.2,'CVIDF1':81.3,'MOTA':78.0}
comparison={k:{'measured':values[k],'paper_target':v,'difference_percentage_points':values[k]-v} for k,v in targets.items()}
primary_gap=max(abs(comparison[k]['difference_percentage_points']) for k in ['HOTA','IDF1','AssA'])
grade=('STRICT PASS' if primary_gap<=0.2 else 'ACCEPTABLE REPRODUCTION' if primary_gap<=0.5 else 'INVESTIGATE' if primary_gap<=1.0 else 'FAIL')
report={'primary_numerical_classification':grade,'primary_max_absolute_gap':primary_gap,'classification_policy':'User project thresholds, not an official paper criterion; numerical closeness does not remove setup deviations','status':'COMPLETE_PENDING_FINAL_AUDIT','metrics_percent':values,'comparison':comparison,'checkpoint_sha256':prepared['inference_completion']['checkpoint_sha256'],'views':len(expected),'scenes':len(scenes),'sources':{'trackeval_detailed':str(detailed),'trackeval_detailed_sha256':h(detailed),'crossview_cvidf1':str(output/'cvidf1/metrics.json'),'crossview_cvma':str(output/'cvma/metrics.json')},'jobs':jobs,'caveats':['User restricted training to GPUs0–3; global batch4 preserves the official per-rank batch1 loader; repository20k+20k recipe.','Documented runtime, mapper, loader-metadata and two-key pretrained compatibility repairs.','Cross-view evaluation runs original metric sources through GNU Octave, not MATLAB.','The two official evaluators ship different GT in ten views; each published GT version is preserved.','TrackEval seqLength metadata alone expanded to1005 for00001garden_View1; GT rows unchanged.'],'test_gt_tuning':False}
(output/'metrics_and_comparison.json').write_text(json.dumps(report,indent=2));(R/'manifests/real_evaluation_complete.json').write_text(json.dumps(report,indent=2))
text='# Real evaluation comparison\n\nActual finalStage2 predictions; no synthetic results included. Full reproduction audit remains required.\n\n| Metric | Measured | Paper | Difference (pp) |\n|---|---:|---:|---:|\n'
for k,v in comparison.items():text+=f"| {k} | {v['measured']:.4f} | {v['paper_target']:.1f} | {v['difference_percentage_points']:+.4f} |\n"
text+='\nPrimary numerical classification: '+grade+'. Maximum absolute primary gap: '+str(primary_gap)+' percentage points. These are user project thresholds, not paper standards. Configuration and runtime deviations remain disclosed regardless of numerical classification.\n'
text+='\n'+ '\n'.join('- '+x for x in report['caveats'])+'\n';(R/'reports/18_real_metric_comparison.md').write_text(text);(R/'reports/VisionTrack_reproduction.md').write_text(text)
print('REAL_EVALUATION_COMPLETE_PENDING_FINAL_AUDIT',flush=True)
