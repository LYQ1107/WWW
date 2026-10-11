"""Official pooled factorial metrics and actual first-commit divergence attribution."""
from jev_phase17_common import *
import argparse,gzip,collections
from run_jev_phase10_closed_loop import metrics
from evaluate_jev_phase13_official_matlab import prepare,MATLAB

def records(p):
 with gzip.open(p,'rt') as stream:return [json.loads(line) for line in stream]
def official(arm):
 folder=OUT/'pooled_factorial'/arm;folder.mkdir(parents=True,exist_ok=True);sources=[];predictions=[];per_video={}
 for v in DEV:
  p=OUT/'factorial/cold'/arm/f'video{v:02d}/RESULT.json';r=read(p);assert r['status']=='COMPLETE';sources.append(ref(p));assert sha(r['raw_predictions']['path'])==r['raw_predictions']['SHA256'];predictions.extend(read(r['raw_predictions']['path']));per_video[str(v)]=r['strict_online_metrics']
 save(folder/'RAW_PREDICTIONS.json',predictions)
 strict,evaluation=metrics(folder/'RAW_PREDICTIONS.json',DEV,folder/'trackeval_raw');manifest=evaluation.parent/'prepared/manifest.json'
 toolkit=Path('/home/liuyeqiang/WWW_jev_phase14/MOTChallengeEvalKit_cv_test/matlab_devkit');old=read(ROOT/'reports/JEV_PHASE16/OFFICIAL_MATLAB_RESULTS.json');assert {str(p.relative_to(toolkit)):sha(p) for p in toolkit.rglob('*') if p.is_file() and p.suffix in ['.m','.cpp','.mexa64']}==old['toolkit_SHA256']
 mf=folder/'official_matlab';jobs,inputs=prepare(manifest,mf);config=mf/'CONFIG.json';native=mf/'NATIVE.json';log=mf/'matlab.log';save(config,dict(kit=str(toolkit),jobs=jobs,output=str(native),benchmark='VisionTrack'))
 env=os.environ.copy();env.update(JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with log.open('w') as stream:subprocess.run([str(MATLAB),'-batch',"run('"+str(ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m')+"')"],env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 data=read(native);identity=[r['Identity'] for r in data['results'] if r['convention']=='sequential'];clear=[r['CLEAR'] for r in data['results'] if r['convention']=='interleaved'];c={k:sum(d[k] for d in identity) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};e={k:sum(d[k] for d in clear) for k in ['fn','fp','id_switches','tp']}
 per_scene={}
 for row in data['results']:
  scene=row['scene'];item=per_scene.setdefault(scene,{})
  if row['convention']=='sequential':i=row['Identity'];item.update(CVIDF1=200*i['IDTP']/(i['n_gt']+i['n_tr']),n_gt=i['n_gt'])
  else:j=row['CLEAR'];item['interleaved_CLEAR_counts']=j
 for item in per_scene.values():j=item['interleaved_CLEAR_counts'];item['CVMA']=100*(1-(j['fn']+j['fp']+j['id_switches'])/item['n_gt'])
 result=dict(status='COMPLETE',binding=binding(),arm=arm,pooled_metrics=strict,per_video=per_video,per_scene_cross_view=per_scene,CVIDF1=200*c['IDTP']/(c['n_gt']+c['n_tr']),CVMA=100*(1-(e['fn']+e['fp']+e['id_switches'])/c['n_gt']),official_report=ref(native),official_toolkit_SHA256=old['toolkit_SHA256'],toolkit_unchanged=True,raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),sources=sources,aggregation='actual pooled TrackEval and sum official per-scene counts',independent_validation=False)
 save(folder/'RESULT.json',result);print('OFFICIAL_FACTORIAL',arm,strict,result['CVIDF1'],result['CVMA'],flush=True);return result

def collect():
 runs={arm:read(OUT/'pooled_factorial'/arm/'RESULT.json') for arm in ['B0','B1','B2','B3']};fields=['HOTA','AssA','IDF1','IDSW','MOTA','Frag'];values={a:dict(r['pooled_metrics'],CVIDF1=r['CVIDF1'],CVMA=r['CVMA']) for a,r in runs.items()};contrasts={name:{k:values[a][k]-values[b][k] for k in fields+['CVIDF1','CVMA']} for name,a,b in [('WHO_given_original_Q4','B1','B0'),('Q4_given_original_WHO','B2','B0'),('WHO_given_safe_Q4','B3','B2'),('Q4_given_safe_WHO','B3','B1'),('joint','B3','B0')]};interaction={k:values['B3'][k]-values['B2'][k]-values['B1'][k]+values['B0'][k] for k in fields+['CVIDF1','CVMA']}
 divergences=[]
 for v in DEV:
  folder=OUT/'factorial/cold';base=read(folder/'B0'/f'video{v:02d}/RESULT.json');baseline={tuple(q['key']):q for q in records(base['commits']['path'])};basequeries={(tuple(q['key']),q['task']):q for q in records(base['queries']['path'])};baseowner={(tuple(q['key']),q['row']):q for q in records(base['owner_trace']['path'])}
  for arm in ['B1','B2','B3']:
   r=read(folder/arm/f'video{v:02d}/RESULT.json');qs={(tuple(q['key']),q['task']):q for q in records(r['queries']['path'])};actualowners=records(r['owner_trace']['path']);first=None;transitions=collections.Counter();changed=0
   for commit in records(r['commits']['path']):
    previous=baseline[tuple(commit['key'])];assert len(previous['ids'])==len(commit['ids'])
    rows=[i for i,(a,b) in enumerate(zip(previous['ids'],commit['ids'])) if a!=b];changed+=len(rows)
    if rows and first is None:
     key=tuple(commit['key']);first=dict(key=list(key),rows=rows,original_IDs=previous['ids'],actual_IDs=commit['ids'],original_events=previous['events'],actual_events=commit['events'],original_query=basequeries.get((key,0)),actual_query=qs.get((key,0)),same_state_local_divergence=r['first_same_state_decision_divergence'])
   for row in actualowners:
    old=baseowner[tuple(row['key']),row['row']];assert row['GT_OFFLINE_ONLY']==old['GT_OFFLINE_ONLY']
    if row['GT_OFFLINE_ONLY'] is None:continue
    left='CORRECT' if old['owner_correct'] else 'WRONG' if old['wrong_owner'] else 'UNKNOWN';right='CORRECT' if row['owner_correct'] else 'WRONG' if row['wrong_owner'] else 'UNKNOWN';transitions[left+'_TO_'+right]+=1
   divergences.append(dict(video=v,arm=arm,first_actual_ID_divergence=first,changed_committed_rows=changed,paired_owner_coverage_transitions=dict(transitions),actual_propagation=read(r['error_propagation']['path']),original_propagation=read(base['error_propagation']['path']),query_source=r['queries'],GT_future_actor_inputs=False,causal_scope='same input, same start and distinct future state; later paths not synchronized'))
 save(REPORTS/'WHO_Q4_FACTORIAL.json',dict(status='COMPLETE_COLD_START',binding=binding(),actual_metrics=values,contrasts=contrasts,interaction_B3_minus_B2_minus_B1_plus_B0=interaction,raw_sources={a:ref(OUT/'pooled_factorial'/a/'RESULT.json') for a in runs},statistics='3 reused DEV videos; descriptive factorial, no independent-test or significance claim',factor_definition={'WHO':'all identity history tensors, metadata and pair evidence, including rule REACT appearance','Q4':'commitment_features only'},cold_reference_matches_XV=values['B0']['IDSW']==147,hard_reference_matches_XVI=values['B3']['IDSW']==1494))
 save(REPORTS/'FIRST_DIVERGENCE.json',dict(status='COMPLETE_COLD_START',binding=binding(),cases=divergences,UNKNOWN_not_correction=True,raw_state_snapshots='first same-state divergence and at most32 detailed captured events per run; all later paths have raw query/commit/admission/cache provenance for reconstruction',snapshots_exhaustive=False))
 print('FACTORIAL_AGGREGATE_COMPLETE',values,contrasts,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B0','B1','B2','B3']);p.add_argument('--collect',action='store_true');a=p.parse_args();protect()
 if a.collect:collect()
 else:official(a.arm)
