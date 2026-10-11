"""Frozen-weight WHO/Q4 factorial, same-state controls and real mutated future."""
from jev_phase17_common import *
import argparse,copy,gzip,collections
import torch
from jev_phase16_common import load_policy
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase15_pilot_native import converted_prefix
from jev_phase15_causal_metrics import FrozenClearFuture
from jev_phase14_artifacts import load_dense
from jev_phase16_memory_risk import MemoryRisk
from gtr.modeling.jev_phase16.native_memory_commit_adapter import attach
from gtr.modeling.jev_phase17.factor_memory import FactorMemory
from gtr.modeling.jev_native_state import fingerprint,prefix_state
from gtr.modeling.jev_stage2.assignment import lawful_choice
from run_jev_phase10_closed_loop import raw_predictions,metrics

ARMS=['B0','B1','B2','B3']
def run_case(video,arm,scope,entry=None,stop=None,live=False,case_name=None):
 protect();assert video in (DEV if scope in ['cold','contract'] else TRAIN);torch.set_num_threads(1)
 torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 values,frames,reader=cache_inputs(video);policy,checkpoint,training=load_policy('v3')
 prefix=converted_prefix(load_dense(entry['prefix']['path'],map_location='cuda:0')) if entry else None
 if entry:assert sha(entry['prefix']['path'])==entry['prefix']['SHA256']
 model=build_tracker(video,policy,react_learned=False,live=live);executor=attach(model,prefix=prefix)
 executor.memory_factory=lambda:FactorMemory(arm,prefix);executor.reset()
 folder=OUT/'factorial'/scope/arm/f'video{video:02d}'
 if entry:folder=folder/f'frame{entry["key"][1]:06d}_view{entry["key"][2]}_row{entry["row"]}'
 if case_name:folder=folder/case_name
 folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
 source=binding(checkpoint=checkpoint,training=training,perception='actual images' if live else 'frozen Stage1 real ROI/ReID cache',seed=20261009,arm=arm,scope=scope,perception_manifest=ref(Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2/stage1_cache_v1')/f'video{video:02d}/RESULT.json'))
 risk=MemoryRisk(video,reader,prefix);journal=gzip.open(folder/'QUERIES.jsonl.gz','wt');commits=gzip.open(folder/'COMMITS.jsonl.gz','wt');admissions=gzip.open(folder/'ADMISSIONS.jsonl.gz','wt')
 start=time.monotonic();pending_native={};startup={};capture_count=[0];counter=collections.Counter();last_progress=[0];first_changed=[None]
 def before_native(**d):
  pending_native.clear();pending_native.update(d)
  if not startup:
   state=prefix_state(model,**d)
   startup.update(native_state_SHA256=fingerprint(state),raw_memory_SHA256=fingerprint(executor.memory.state_dict()['raw']),RNG_SHA256=fingerprint({k:v for k,v in state.items() if 'rng' in k}))
 model.jev_native_prefix_observer=before_native
 def before(**d):
  risk.before(**d);comparison=executor.memory.last_comparison;raw=comparison['raw'];safe=comparison['safe'];task=d['task'];key=[d['context']['frame'],d['context']['view']]
  raw=dict(raw);raw['question_type']=d['batch']['question_type'].clone()
  details=getattr(policy,'runtime_details',None)
  if task==0:base=policy(raw)[0]
  else:
   v=raw['pair_evidence'][0,:,:,:3].max(-1).values;base=torch.cat([v,v.new_full((len(v),1),.75)],-1)
  policy.runtime_details=details
  actual=d['logits'];original=lawful_choice(base,raw['legal'][0]);executed=lawful_choice(actual,d['batch']['legal'][0]);changed=[i for i,(a,b) in enumerate(zip(original,executed)) if a!=b]
  counter['queries']+=1;counter['same_state_changed_assignments']+=len(changed)
  counter['Q4_changed_queries']+=int(not torch.equal(raw['commitment_features'],safe['commitment_features']))
  counter['WHO_changed_queries']+=int(not torch.equal(raw['history_visual'],safe['history_visual']))
  fallback=[]
  for ref_id in d['refs']:
   indices,flag=executor.memory.admission.matching_indices(ref_id)
   if flag or not indices:fallback.append({'identity':ref_id,'reason':'pending_without_accepted' if indices else 'last_quarantined_real_observation','indices':indices or [len(executor.memory.raw_galleries[ref_id])-1]})
  row=dict(key=key,task=task,refs=d['refs'],rows=d['context'].get('rows',list(range(len(actual)))),original_logits=base.cpu().tolist(),actual_logits=actual.cpu().tolist(),original_assignment=original,actual_assignment=executed,changed_rows=changed,WHO_raw_SHA256=fingerprint({k:raw[k] for k in ['history_visual','history_mask','identity_meta','pair_evidence']}),WHO_safe_SHA256=fingerprint({k:safe[k] for k in ['history_visual','history_mask','identity_meta','pair_evidence']}),Q4_raw_SHA256=fingerprint(raw['commitment_features']),Q4_safe_SHA256=fingerprint(safe['commitment_features']),Q4_max_abs_delta=float((raw['commitment_features']-safe['commitment_features']).abs().max()) if raw['commitment_features'].numel() else 0,fallback=fallback)
  if (changed or counter['queries']==1) and capture_count[0]<32:
   n=capture_count[0];p=folder/f'event{n:03d}.pt';torch.save({'raw_batch':raw,'safe_batch':safe,'actual_batch':d['batch'],'raw_memory':executor.memory.state_dict()['raw'],'trusted_memory':executor.memory.trusted.state_dict(),'admission':executor.memory.admission.state_dict(),'Gallery':executor.memory.raw_galleries},p);row['source_state']=ref(p);capture_count[0]+=1
   if changed and first_changed[0] is None:
    state=prefix_state(model,**pending_native);state['phase13_identity_meta']=executor.memory.state_dict();sp=folder/'FIRST_LOCAL_DIVERGENCE_NATIVE.pt';torch.save(state,sp);row['first_divergence_native']=ref(sp);row['first_divergence_fingerprint']=fingerprint(state);first_changed[0]=copy.deepcopy(row)
  journal.write(json.dumps(row,allow_nan=False)+'\n')
 def after(**d):
  d['video']=video;risk.after(**d);ids=d['instances'][-1].track_ids.cpu().tolist()
  commits.write(json.dumps(dict(key=[d['frame'],d['view']],ids=ids,events=d['events']))+'\n')
  for event in executor.memory.last_admission:admissions.write(json.dumps(dict(key=[d['frame'],d['view']],**event),allow_nan=False)+'\n')
  if time.monotonic()-last_progress[0]>=15:
   save(folder/'PROGRESS.json',dict(status='RUNNING',key=[d['frame'],d['view']],frames=frames,arm=arm,scope=scope,seconds=time.monotonic()-start,counts=dict(counter)));journal.flush();commits.flush();admissions.flush();last_progress[0]=time.monotonic()
 executor.observer=before;executor.commit_observer=after
 boundary=tuple(entry['key'][1:]) if entry else (0,0)
 end=min(frames-1,boundary[0]+31) if entry else stop
 try:
  with torch.no_grad():raw_result,_=run(model,values,frames,stop=end,prefix=copy.deepcopy(prefix))
 finally:journal.close();commits.close();admissions.close()
 predictions=raw_predictions(raw_result,risk.labels.images);save(folder/'RAW_PREDICTIONS.json',predictions)
 evaluation=None;strict=None;H=None
 if scope=='cold':strict,evaluation=metrics(folder/'RAW_PREDICTIONS.json',[video],folder/'strict_eval')
 elif entry:
  metric=FrozenClearFuture('multi_question',20261009,video,case_override=XV/'train_commitment_prefixes_v1'/f'video{video:02d}',duplicate_gt_diagnostic=video==14)
  actual_ids=metric.ids(raw_result);H={str(h):dict(**metric.horizon(actual_ids,boundary,h,frames),identity_consequences=risk.horizon(boundary,min(frames,boundary[0]+h))) for h in [8,16,32]}
 end_frame=(frames if end is None else end+1);error=risk.horizon(boundary,end_frame)
 save(folder/'ERROR_PROPAGATION.json',error)
 with gzip.open(folder/'OWNER_TRACE.jsonl.gz','wt') as stream:
  for row in risk.strong_trace:stream.write(json.dumps(row)+'\n')
 result=dict(status='COMPLETE',binding=source,video=video,arm=arm,scope=scope,frames=frames,processed_end=end_frame,startup=startup,prefix=entry,raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),strict_online_metrics=strict,evaluator_result=ref(evaluation/'metrics.json') if evaluation else None,H=H,diagnostic_counts=dict(counter),first_same_state_decision_divergence=first_changed[0],queries=ref(folder/'QUERIES.jsonl.gz'),commits=ref(folder/'COMMITS.jsonl.gz'),admission_trace=ref(folder/'ADMISSIONS.jsonl.gz'),owner_trace=ref(folder/'OWNER_TRACE.jsonl.gz'),error_propagation=ref(folder/'ERROR_PROPAGATION.json'),final_raw_memory_SHA256=fingerprint(executor.memory.state_dict()['raw']),seconds=time.monotonic()-start,GT_future_actor_inputs=False,actual_mutated_state=True,latency_valid=False)
 save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status='COMPLETE',result=ref(folder/'RESULT.json')))
 print('PHASE17_CASE_COMPLETE',video,arm,scope,boundary,strict,flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--scope',choices=['cold','prefix','contract'],required=True);p.add_argument('--stop',type=int);p.add_argument('--case-name');a=p.parse_args()
 if a.scope!='prefix':run_case(a.video,a.arm,a.scope,stop=a.stop,case_name=a.case_name);return
 manifest=read(XV/'train_commitment_prefixes_v1'/f'video{a.video:02d}/RESULT.json');results=[]
 for entry in manifest['counterfactual_prefixes']:results.append(run_case(a.video,a.arm,a.scope,entry=entry))
 save(OUT/'factorial/prefix'/a.arm/f'video{a.video:02d}/RESULT.json',dict(status='COMPLETE',binding=binding(),cases=results))
if __name__=='__main__':main()
