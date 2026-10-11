"""Reconstruct full native state, fork real H64 futures, restore original replay."""
from jev_phase18_common import *
import argparse,gzip,io,copy,collections
import torch
from jev_phase16_common import load_policy
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase16_memory_risk import MemoryRisk
from jev_phase18_metrics import AuditedClearFuture
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_native_state import prefix_state,restore_state,fingerprint
from gtr.modeling.jev_phase18.constraint import NativeConstraint

def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def main(video,resume=False):
 protect();assert video in TRAIN;assert read(REPORTS/'NATIVE_CONSTRAINT_CONTRACT.json')['status']=='PASS_REAL_NATIVE_PARITY'
 torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 selected=[e for e in read(OUT/'selection/INTERVENTION_EVENTS.json') if e['event_key'][0]==video];assert selected
 census=read(OUT/'census_v2'/f'video{video:02d}/RESULT.json');commits={tuple(e['key']):e for e in rows(census['commits']['path'])};queries={(tuple(e['key']),e['task']):e for e in rows(census['queries']['path'])}
 values,frames,reader=cache_inputs(video);policy,checkpoint,_=load_policy('v3');model=build_tracker(video,policy,react_learned=False);executor=attach(model);branch_model=[None];branch_executor=[None]
 metric=AuditedClearFuture('multi_question',20261009,video,case_override=Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1/train_commitment_prefixes_v1')/f'video{video:02d}')
 original_folder=OUT/'interventions'/f'video{video:02d}'
 folder=(OUT/'interventions_v2'/f'video{video:02d}') if resume else original_folder
 folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists();start=time.monotonic();results=[];captured=set();replay_count=[0];latest=[0]
 if resume:
  assert video==16,'only actual failed video; original successful results immutable'
  for e in selected:
   old=original_folder/f'PREFIX_{e["selection_index"]:02d}.json'
   paths=[original_folder/f'event{e["selection_index"]:02d}'/v/'RESULT.json' for v in ['BASE']+[f'ALT{n}' for n in range(len(e['alternatives_GT_free']))]]
   if old.exists() and all(p.exists() and read(p)['status']=='COMPLETE' for p in paths):
    captured.add(e['selection_index']);results.extend(ref(p) for p in paths)
 def branch(event,blob,initial,variant,reference):
  prefix=torch.load(io.BytesIO(blob),map_location='cuda:0');assert fingerprint(prefix)==initial
  bm=branch_model[0];be=branch_executor[0];risk=MemoryRisk(video,reader,prefix);trace=[];first=[];support={};boundary=tuple(event['event_key'][1:3]);row=event['event_key'][3];stop=min(frames-1,boundary[0]+63)
  def before(**d):
   risk.before(**d);key=tuple((d['context']['frame'],d['context']['view']))
   if d['task']==0 and key==boundary:
    record=dict(refs=d['refs'],scores=d['logits'].cpu().tolist(),legal=d['batch']['legal'][0].cpu().tolist());expected=queries[key,0];assert record=={k:expected[k] for k in record},'restored first scores/refs/legal differ';first.append(record)
   if d['task']==0:
    gts=risk.labels.current(*key);legal=d['batch']['legal'][0].cpu().tolist()
    for r,gt in enumerate(gts):support[key,r]=gt is not None and any(legal[r][j] and risk.owner.get(ref)==gt for j,ref in enumerate(d['refs']))
   else:
    gts=risk.labels.current(*key);legal=d['batch']['legal'][0].cpu().tolist()
    for local,r in enumerate(d['context']['rows']):support[key,r]|=gts[r] is not None and any(legal[local][j] and risk.owner.get(ref)==gts[r] for j,ref in enumerate(d['refs']))
  def after(**d):
   d['video']=video;risk.after(**d);key=(d['frame'],d['view']);ids=d['instances'][-1].track_ids.cpu().tolist();trace.append(dict(key=list(key),ids=ids,events=d['events']))
   for r in risk.strong_trace[-len(ids):] if ids else []:r['false_split_birth_anchored']=r['action']=='START_NEW' and bool(support.get((key,r['row'])))
   if variant=='BASE':assert ids==commits[key]['ids'],'paired base continuation is not actual original trajectory'
  be.observer=before;be.commit_observer=after
  if variant=='BASE':
   with torch.no_grad():raw,_=run(bm,values,frames,stop=stop,prefix=prefix)
   audit=None
  else:
   with NativeConstraint(be,boundary,row,reference) as constraint,torch.no_grad():raw,_=run(bm,values,frames,stop=stop,prefix=prefix)
   assert constraint.fired==1;assert reference is None or trace[0]['ids'][row]==reference;audit=constraint.audit
  ids=metric.ids(raw);horizons={}
  for h in [16,32,64]:
   consequence=risk.horizon(boundary,min(frames,boundary[0]+h));consequence['counts']['false_split_birth_anchored']=sum(r['false_split_birth_anchored'] for r in risk.strong_trace if boundary<=tuple(r['key']) and r['key'][0]<boundary[0]+h);horizons[str(h)]=dict(**metric.horizon(ids,boundary,h,frames),identity_consequences=consequence)
  out=folder/f'event{event["selection_index"]:02d}'/variant;out.mkdir(parents=True,exist_ok=True)
  for name,data in [('COMMITS',trace),('OWNER_TRACE',risk.strong_trace)]:
   with gzip.open(out/(name+'.jsonl.gz'),'wt') as f:
    for item in data:f.write(json.dumps(item)+'\n')
  result=dict(status='COMPLETE',event_key=event['event_key'],group=event['group'],split=event['split'],category=event['sampled_stratum'],variant=variant,reference=reference,starting_native_state_fp=initial,first_scores_exact=True,H=horizons,constraint=audit,commits=ref(out/'COMMITS.jsonl.gz'),owner_trace=ref(out/'OWNER_TRACE.jsonl.gz'),actual_mutated_state=True,GT_actor_input=False)
  save(out/'RESULT.json',result);results.append(ref(out/'RESULT.json'));print('NATIVE_BRANCH_COMPLETE',video,event['selection_index'],variant,flush=True);del raw,prefix,risk;be.memory=be.memory_factory();be.latency=[];torch.cuda.empty_cache()
 def capture(**d):
  key=(d['frame'],d['view']);wanted=[e for e in selected if tuple(e['event_key'][1:3])==key and e['selection_index'] not in captured]
  if not wanted:return
  state=prefix_state(model,**d);state['phase13_identity_meta']=executor.memory.state_dict();initial=fingerprint(state);buf=io.BytesIO();torch.save(state,buf);blob=buf.getvalue();digest=hashlib.sha256(blob).hexdigest();saved_prefix=None
  prefix_paths=[p for directory in ['prefixes','prefixes_v2'] for p in (OUT/directory).rglob('*.pth')]
  persistent=sum(p.stat().st_size for p in prefix_paths)
  # Disjoint 600MiB per-video reservations prevent a shared storage-cap race.
  if persistent+len(blob)<=2*2**30 and sum(p.stat().st_size for p in prefix_paths if p.parent.name==f'video{video:02d}')+len(blob)<=600*2**20:
   path=OUT/('prefixes_v2' if resume else 'prefixes')/f'video{video:02d}'/f'frame{key[0]:06d}_view{key[1]}.pth';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();path.write_bytes(blob);saved_prefix=ref(path)
  if branch_model[0] is None:branch_model[0]=build_tracker(video,policy,react_learned=False);branch_executor[0]=attach(branch_model[0])
  try:
   for e in wanted:
    branch(e,blob,initial,'BASE',None)
    for n,reference in enumerate(e['alternatives_GT_free']):branch(e,blob,initial,f'ALT{n}',reference)
    captured.add(e['selection_index']);save(folder/f'PREFIX_{e["selection_index"]:02d}.json',dict(event_key=e['event_key'],actual_native_fingerprint=initial,serialized_bytes=len(blob),serialized_SHA256=digest,persistent_prefix=saved_prefix,otherwise='RAM serialized exact native state; reconstructable from frozen checkpoint/cache and verified full original replay; no fabricated lightweight mirror',source_census=ref(OUT/'census_v2'/f'video{video:02d}/RESULT.json')))
  finally:
   restore_state(model,state);executor.memory.load_state_dict(state['phase13_identity_meta']);assert fingerprint(prefix_state(model,**d))==fingerprint({k:v for k,v in state.items() if k!='phase13_identity_meta'});assert fingerprint(executor.memory.state_dict())==fingerprint(state['phase13_identity_meta'])
  save(folder/'PROGRESS.json',dict(status='RUNNING',captured_groups=len(captured),total_groups=len(selected),completed_branches=len(results),last_key=list(key),seconds=time.monotonic()-start,pid=os.getpid()));del buf,blob,state
 def before(**d):
  e=queries[(d['context']['frame'],d['context']['view']),d['task']];assert d['refs']==e['refs'] and d['logits'].cpu().tolist()==e['scores'] and d['batch']['legal'][0].cpu().tolist()==e['legal'],'original replay drift'
 def after(**d):
  key=(d['frame'],d['view']);e=commits[key];inst=d['instances'][-1];assert inst.track_ids.cpu().tolist()==e['ids'] and fingerprint(inst.reid_features)==e['committed_vectors_SHA'] and fingerprint(executor.memory.state_dict())==e['raw_memory_SHA'],'original replay/native state drift';replay_count[0]+=1
  if time.monotonic()-latest[0]>15:
   save(folder/'PROGRESS.json',dict(status='RUNNING',key=list(key),captured_groups=len(captured),total_groups=len(selected),completed_branches=len(results),seconds=time.monotonic()-start,pid=os.getpid()));latest[0]=time.monotonic()
 model.jev_native_prefix_observer=capture;executor.observer=before;executor.commit_observer=after
 with torch.no_grad():run(model,values,frames,stop=max(e['event_key'][1] for e in selected))
 assert len(captured)==len(selected);save(folder/'RESULT.json',dict(status='COMPLETE',binding=binding(checkpoint=checkpoint),video=video,groups=len(selected),branches=results,resumed_completed_groups_without_reexecution=resume,actual_duplicate_GT_timesteps=metric.duplicates,full_original_replay_verified_payloads=replay_count[0],forks_and_restores_exact=True,actual_prefixes='full Instances/Gallery/hits/Bank/RNG plus XV raw commitment state, optional persistent bounded snapshots',seconds=time.monotonic()-start));save(folder/'PROGRESS.json',dict(status='COMPLETE',groups=len(selected),branches=len(results)));print('NATIVE_INTERVENTIONS_COMPLETE',video,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--resume-failed-video16',action='store_true');a=p.parse_args();main(a.video,a.resume_failed_video16)
