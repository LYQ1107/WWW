"""Full native mutated-state online comparison; GT is evaluated after inference."""
import argparse,gzip,json,time,collections
from pathlib import Path
import numpy as np
import torch
from jev_phase10_common import *
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_candidate_assignment import assign_candidate_values
from gtr.modeling.jev_native_state import prefix_state,fingerprint,NativeStateForkAdapter


def policy(case):
 if case['kind']=='off':return None
 if case['kind']=='model':
  p=case['checkpoint'];assert sha(p['path'])==p['SHA256'];return CandidateValuePolicy('model_fixed_new',torch.jit.load(p['path'],map_location='cuda:0').eval())
 return CandidateValuePolicy(case['policy'],dynamic_alpha=case.get('alpha',0.))


def weight_bytes(p):
 return 0 if p is None or p.model is None else sum(t.numel()*t.element_size()for t in list(p.model.parameters())+list(p.model.buffers()))


def metadata(video):
 if video in HELDOUT:heldout_guard(video)
 else:assert video in VAL
 ann=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text());images=[i for i in ann['images']if int(i['video_id'])==video];lookup={(int(i['frame_id'])-1,int(i['view_id'])-1):i for i in images};frames=max(f for f,v in lookup)+1;assert len(lookup)==2*frames
 inputs=[{'video_id':video,'view_num':2,'height':int(lookup[f,v]['height']),'width':int(lookup[f,v]['width']),'image':None}for v in range(2)for f in range(frames)]
 return inputs,lookup,frames


def raw_predictions(instances,images):
 output=[]
 for index,instance in enumerate(instances):
  f,v=divmod(index,2);im=images[f,v];h,w=instance.image_size;sx=im['width']/w;sy=im['height']/h
  for box,t,score in zip(instance.pred_boxes.tensor.detach().cpu().tolist(),instance.track_ids.detach().cpu().tolist(),instance.scores.detach().cpu().tolist()):
   x,y,x2,y2=box;output.append({'image_id':im['id'],'track_id':int(t),'bbox':[x*sx,y*sy,max(0.,(x2-x)*sx),max(0.,(y2-y)*sy)],'score':float(score),'category_id':1})
 return output


def metrics(prediction_path,videos,out):
 # These helpers operate on an isolated TRAIN subset and output root. The
 # internal TrackEval split label "test" is not the official TEST dataset.
 import run_early_pilot_tracking as pilot
 ann=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text());ims=[i for i in ann['images']if int(i['video_id'])in videos];ids={i['id']for i in ims};subset={**ann,'images':ims,'annotations':[a for a in ann['annotations']if a['image_id']in ids],'videos':[v for v in ann['videos']if int(v['id'])in videos]};pilot.PILOT=out;dataset=pilot.prepare_eval_dataset(subset);_,evaluated=pilot.run_eval('native',prediction_path,dataset);return pilot.extract_metrics(evaluated),evaluated


def paired_h32(model,actual,allinputs,images,evaluator,aligned,commits,prefixpath,out):
 from jev_phase8_opportunity import PrefixIdentityAnchors
 from jev_phase8_utility import effects
 anchors=PrefixIdentityAnchors();groups=collections.defaultdict(dict)
 for (frame,view,row),r in sorted(aligned.items()):
  if(frame,view)<(128,0):groups[frame,view][row]=r
 for (frame,view),rs in sorted(groups.items()):anchors.update({row:r['id']for row,r in rs.items()},{row:r['gt']for row,r in rs.items()},frame)
 expected={tuple(d['key'][1:]):d for d in commits};branches={};fixed=CandidateValuePolicy('gmt_values');model.jev_candidate_latency_observer=None
 for tag in ['actual','same_prefix_Fixed_current']:
  traces=[]
  def before(**d):
   model.jev_candidate_policy=fixed if tag!='actual'and(d['frame'],d['view'])==(128,0)else actual;model.__dict__.pop('_jev_candidate_last',None)
  def after(**d):
   inst=d['instances'][-1];ids=inst.track_ids.detach().cpu().tolist();assert len(set(ids))==len(ids);record={'key':[int(d['frame']),int(d['view'])],'ids':ids,'id_count':int(d['id_count']),'hits':{str(t):int(d['hits'][t])for t in ids},'gallery_lengths':{str(t):len(d['galleries'][t])for t in ids}};traces.append(record)
   if tag=='actual':
    original=expected[tuple(record['key'])]
    assert all(record[k]==original[k]for k in ['ids','id_count','hits','gallery_lengths']),'paired actual H32 differs from complete online path'
  model.jev_native_prefix_observer=before;model.jev_candidate_commit_observer=after
  with torch.no_grad():raw,_=NativeStateForkAdapter(model).run(prefixpath,allinputs,stop_frame=159)
  preds=[p for p in raw_predictions(raw,images)if 128<=int(evaluator.images[p['image_id']]['frame_id'])-1<=159];save(out/f'PAIRED_H32_{tag}_PREDICTIONS.json',preds);observed=evaluator.align(preds)
  branches[tag]={'effects':{str(h):effects(observed,anchors.diagnostics(),128,h,set())for h in [8,16,32]},'native_commits':traces,'predictions_SHA256':sha(out/f'PAIRED_H32_{tag}_PREDICTIONS.json')}
 delta={str(h):{k:branches['actual']['effects'][str(h)][k]-branches['same_prefix_Fixed_current']['effects'][str(h)][k]for k in ['utility','utility_birth_zero','wrong_identity_duration_camera_frames']}for h in [8,16,32]}
 result={'status':'PASS','prefix_SHA256':sha(prefixpath),'fixed_boundary':[128,0],'branches':branches,'actual_minus_same_prefix_Fixed':delta,'H32_regret_against_executed_Fixed_alternative':max(0,-delta['32']['utility']),'actual_current_and_future_matches_complete_online_path':True,'same_future_policy':'actual frozen controller, freshly computed on each mutated branch','scope':'paired whole-current-payload legal assignment intervention at a predeclared GT-independent boundary; one boundary per video/controller. Regret only against this executed alternative, not an oracle over all candidates.','no_GT_actor_input':True};save(out/'PAIRED_H32_AUDIT.json',result);return {k:v for k,v in result.items()if k!='branches'}|{'branches_effects':{tag:b['effects']for tag,b in branches.items()}}


def run(video,name,phase):
 protect();jit_runtime=configure_candidate_torchscript();source=binding();assert not source['dirty'];assert phase in ['validation','heldout']
 if phase=='heldout':protocol=heldout_guard(video);assert video in HELDOUT
 else:protocol=json.loads((REPORTS/'PHASE10_VALIDATION_CONTROLLER_PROTOCOL.json').read_text());assert video in VAL and protocol['status']=='FROZEN_BEFORE_VALIDATION_CLOSED_LOOP'
 case=next(c for c in protocol['cases']if c['name']==name);out=OUT/f'{phase}_closed_loop_v1'/name/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True);done=out/'RESULT.json'
 if done.exists():assert json.loads(done.read_text())['binding']==source;print('CLOSED_LOOP_ALREADY_COMPLETE',phase,name,video);return
 torch.manual_seed(20261008);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;model=build_model(video,allow_heldout=phase=='heldout');actual=policy(case);reference=policy(protocol.get('latency_reference',{'kind':'rule','policy':'gmt_values'}));model.jev_candidate_policy=actual;rows,images,frames=metadata(video);save(out/'START.json',{'binding':source,'case':case,'frames':frames,'phase':phase,'trajectory_policy':'native mutable production on every future frame','no_GT_actor_inputs':True,'JIT_runtime':jit_runtime})
 latencies=[];events=[];counts=collections.Counter();state=[None];lastref=[0.];journal=None;commitstream=None;stamp=[0.]
 if actual is not None and actual.model is not None:
  with torch.no_grad():
   for _ in range(3):actual.model(torch.zeros(2,64,device='cuda:0'),torch.zeros(2,16,12,device='cuda:0'),torch.ones(2,16,dtype=torch.bool,device='cuda:0'))
 def before(**d):
  start=max(0,d['frame']+1-model.test_len)*2;end=d['frame']*2+d['view'];active=[i.track_ids for i in d['instances'][start:end]if i.has('track_ids')];active=set(torch.cat(active).detach().cpu().tolist())if active else set()
  state[0]={'key':[video,d['frame'],d['view']],'id_count':int(d['id_count']),'gallery_lengths':{int(t):len(g)for t,g in d['galleries'].items()},'active_ids':active,'begin':time.perf_counter()};model.__dict__.pop('_jev_candidate_last',None)
  if(d['frame'],d['view'])==(128,0):
   path=out/'CAUSAL_PREFIX_128_0.pth';current=prefix_state(model,**d)
   if path.exists():assert fingerprint(torch.load(path,map_location=model.device))==fingerprint(current),'resumed closed-loop causal prefix differs'
   else:
    tmp=path.with_suffix('.pth.tmp');torch.save(current,tmp);tmp.replace(path)
 def profile(**d):
  batch=d.pop('batch');torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();v,n=reference.score(batch);assign_candidate_values(v,n,batch.candidate_ids,batch.legal_mask,mode=reference.assignment_mode,legacy_thresholds=batch.thresholds);torch.cuda.synchronize();refms=(time.perf_counter()-begin)*1000;peak=torch.cuda.max_memory_allocated()-allocated;lastref[0]=refms;d.update(reference_policy_assignment_ms=refms,extra_policy_assignment_ms=d['policy_assignment_ms']-refms,extra_peak_VRAM_MiB=(weight_bytes(actual)-weight_bytes(reference)+d['peak_temporary_bytes']-peak)/(1024**2));latencies.append(d)
 def after(**d):
  before=state[0];inst=d['instances'][-1];ids=inst.track_ids.detach().cpu().tolist();assert len(set(ids))==len(ids);counts['payloads']+=1;counts['detections']+=len(ids);batch=d['candidate'];key=[video,d['frame'],d['view']]
  lifecycle=[{'birth':t>before['id_count'],'write':len(d['galleries'][t])>before['gallery_lengths'].get(t,0),'bank_reactivation':t<=before['id_count']and t not in before['active_ids']}for t in ids]
  commit={'key':key,'ids':ids,'id_count':int(d['id_count']),'hits':{int(t):int(d['hits'][t])for t in ids},'gallery_lengths':{int(t):len(d['galleries'][t])for t in ids},'possible_ids':sorted(d['possible_ids']),'old_ids':sorted(d['old_ids']),'trajectory_draw_count':len(d['trajectory_rng']._trajectory_draws),'events':lifecycle};commitstream.write(json.dumps(commit)+'\n')
  for row,t in enumerate(ids):
   if t>before['id_count']:counts['births']+=1
   elif len(d['galleries'][t])>before['gallery_lengths'].get(t,0):counts['memory_writes']+=1
   else:counts['memory_skips']+=1
   counts['bank_reactivations']+=lifecycle[row]['bank_reactivation']
  if batch is not None:
   b=batch['batch'];values=batch['candidate_values'].detach().cpu();raw=b.scores.detach().cpu();mask=b.legal_mask.detach().cpu();proposal=[]
   for row in range(len(raw)):
    cols=torch.where(b.evidence12[row,:,2].detach().cpu()>0)[0];proposal.append(b.candidate_ids[int(cols[0])]if len(cols)else None)
   journal.write(json.dumps({'key':key,'candidate_ids':b.candidate_ids,'initial_proposal_ids':proposal,'GMT_scores':raw.tolist(),'thresholds':b.thresholds.detach().cpu().tolist(),'candidate_values':values.tolist(),'legal_mask':mask.tolist(),'existing_ids':batch['assignment'].existing_ids,'NEW_rows':batch['assignment'].new_rows,'committed_ids':ids})+'\n');counts['semantic_NEW_choices']+=len(batch['assignment'].new_rows);counts['existing_choices']+=sum(t>=0 for t in batch['assignment'].existing_ids)
  elapsed=(time.perf_counter()-before['begin'])*1000-lastref[0]if batch is not None else(time.perf_counter()-before['begin'])*1000;events.append({'key':key,'native_association_and_commit_ms_excluding_paired_reference':elapsed})
  if time.monotonic()-stamp[0]>20:save(out/'PROGRESS.json',{'status':'RUNNING','last_key':key,'frames':frames,'native_payloads':counts['payloads'],'births':counts['births'],'source_commit':source['source_commit']});stamp[0]=time.monotonic()
 model.jev_native_prefix_observer=before;model.jev_candidate_commit_observer=after
 if actual is not None:model.jev_candidate_latency_observer=profile
 begin=time.monotonic()
 resume=out/'RUN_RESUME.json';parts=[];start=0;prefix=None
 if resume.exists():
  saved=json.loads(resume.read_text());assert saved['binding']==source and saved['case']==case;assert sha(saved['prefix']['path'])==saved['prefix']['SHA256'];prefix=saved['prefix']['path'];start=saved['next_frame'];parts=saved['parts'];counts.update(saved['counts']);latencies.extend(saved['latencies']);events.extend(saved['events'])
 class SegmentBoundary(Exception):pass
 chunkroot=out/'chunks';chunkroot.mkdir(exist_ok=True)
 while start<frames:
  stop=min(frames,start+256);attempt=0
  while(chunkroot/f'frame{start:06d}_attempt{attempt:03d}').exists():attempt+=1
  part=chunkroot/f'frame{start:06d}_attempt{attempt:03d}';part.mkdir();journal=gzip.open(part/'CANDIDATE_JOURNAL.jsonl.gz','wt');commitstream=gzip.open(part/'NATIVE_COMMITS.jsonl.gz','wt');nextprefix=[None]
  def segmented_before(**d):
   if d['frame']==stop and d['view']==0:
    # Alternating mutable recovery slots preserve the preceding valid slot
    # until the new atomic manifest commits. They are not frozen evidence.
    path=out/f'UNFROZEN_RECOVERY_SLOT_{(stop//256)%2}.pth';tmp=path.with_suffix('.pth.tmp');torch.save(prefix_state(model,**d),tmp);tmp.replace(path);nextprefix[0]={'path':str(path),'SHA256':sha(path)};raise SegmentBoundary()
   before(**d)
  model.jev_native_prefix_observer=segmented_before
  try:
   with torch.no_grad():
    if prefix is None:instances,_=model.sliding_inference_GMT(rows,2,[None,None,list(range(len(rows)))],native_raw=True)
    else:instances,_=NativeStateForkAdapter(model).run(prefix,rows,stop_frame=frames-1)
   finished=True
  except SegmentBoundary:finished=False
  finally:journal.close();commitstream.close()
  parts.append({'start_frame':start,'stop_exclusive':stop,'journal':str(part/'CANDIDATE_JOURNAL.jsonl.gz'),'journal_SHA256':sha(part/'CANDIDATE_JOURNAL.jsonl.gz'),'commits':str(part/'NATIVE_COMMITS.jsonl.gz'),'commits_SHA256':sha(part/'NATIVE_COMMITS.jsonl.gz')});save(part/'COMPLETE.json',{'status':'COMPLETE','binding':source,'part':parts[-1],'total_committed_payloads':counts['payloads']})
  if finished:assert stop==frames;break
  assert nextprefix[0]is not None;save(resume,{'status':'RESUMABLE','binding':source,'case':case,'next_frame':stop,'prefix':nextprefix[0],'parts':parts,'counts':dict(counts),'latencies':latencies,'events':events});prefix=nextprefix[0]['path'];start=stop;print('NATIVE_SEGMENT_COMMITTED',phase,name,video,start,frames,flush=True)
 # Gzip supports concatenated members; retain each immutable completed part
 # and every failed attempt. No compressed stream is truncated on resume.
 for key,filename in [('journal','CANDIDATE_JOURNAL.jsonl.gz'),('commits','NATIVE_COMMITS.jsonl.gz')]:
  target=out/filename;tmp=target.with_suffix('.gz.tmp')
  with tmp.open('wb')as handle:
   for part in parts:
    assert sha(part[key])==part[key+'_SHA256'];handle.write(Path(part[key]).read_bytes())
  tmp.replace(target)
 rawpred=raw_predictions(instances,images);save(out/'RAW_PREDICTIONS.json',rawpred);save(out/'NATIVE_SEGMENTS.json',{'status':'COMPLETE','binding':source,'frames':frames,'parts':parts,'recovery':'lossless native prefix with ordered gallery/history/bank/RNG; 256-frame segments, alternating mutable recovery slots, atomic cursor; failed attempt logs retained','native_payloads':counts['payloads']});assert counts['payloads']==frames*2
 # Exactly the existing production final filtering and postprocess semantics.
 from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
 filtered=model._remove_short_track(instances)if model.min_track_len>0 else instances
 if model.roi_heads.delay_cls:filtered=model._delay_cls(filtered,video_id=video)
 batchinputs=[rows[v*frames+f]for f in range(frames)for v in range(2)];processed=CustomRCNN._postprocess(filtered,batchinputs,[(0,0)]*len(batchinputs),not_clamp_box=model.not_clamp_box);pred=[]
 for index,value in enumerate(processed):
  f,v=divmod(index,2);im=images[f,v];inst=value['instances']
  for box,t,score in zip(inst.pred_boxes.tensor.detach().cpu().tolist(),inst.track_ids.detach().cpu().tolist(),inst.scores.detach().cpu().tolist()):
   x,y,x2,y2=box;pred.append({'image_id':im['id'],'track_id':int(t),'bbox':[x,y,max(0.,x2-x),max(0.,y2-y)],'score':float(score),'category_id':1})
 p=out/'PREDICTIONS.json';save(p,pred);m,evaluated=metrics(p,[video],out)
 from jev_phase7_offline import IdentityEvaluator
 evaluator=IdentityEvaluator(video);aligned=evaluator.align(rawpred);identity=evaluator.summarize(aligned);save(out/'IDENTITY_METRICS.json',identity);save(out/'LATENCY_SAMPLES.json',{'policy':latencies,'native_steps':events})
 from jev_phase10_online_audit import audit
 diagnostic=audit(aligned,out/'CANDIDATE_JOURNAL.jsonl.gz',out/'NATIVE_COMMITS.jsonl.gz');save(out/'ONLINE_IDENTITY_AUDIT.json',diagnostic)
 with gzip.open(out/'NATIVE_COMMITS.jsonl.gz','rt')as handle:commits=list(map(json.loads,handle))
 causal=paired_h32(model,actual,rows,images,evaluator,aligned,commits,out/'CAUSAL_PREFIX_128_0.pth',out)if frames>=160 else {'status':'CENSORED_VIDEO_TOO_SHORT_FOR_PREDECLARED_BOUNDARY'}
 def quantile(field):
  vals=[d[field]for d in latencies[10:]];return dict(zip(['p50','p95','max'],map(float,np.quantile(vals,[.5,.95,1]))))if vals else None
 result={'status':'COMPLETE','binding':source,'phase':phase,'case':case,'video':video,'frames':frames,'counts':dict(counts),'metrics':m,'predictions':{'path':str(p),'SHA256':sha(p)},'raw_predictions':{'path':str(out/'RAW_PREDICTIONS.json'),'SHA256':sha(out/'RAW_PREDICTIONS.json')},'journal':{'path':str(out/'CANDIDATE_JOURNAL.jsonl.gz'),'SHA256':sha(out/'CANDIDATE_JOURNAL.jsonl.gz')},'native_commits_SHA256':sha(out/'NATIVE_COMMITS.jsonl.gz'),'identity_summary':{k:v for k,v in identity.items()if k not in ['wrong_ID_episodes','identity_index_query']},'identity_index_query':{k:v for k,v in identity['identity_index_query'].items()if k!='queries'},'online_candidate_lifecycle_audit':{k:v for k,v in diagnostic.items()if k not in ['wrong_ID_episodes','rows','row_statuses']},'paired_H32_causal_audit':causal,'policy_latency_ms':quantile('policy_assignment_ms'),'extra_paired_latency_ms':quantile('extra_policy_assignment_ms'),'extra_peak_VRAM_MiB':quantile('extra_peak_VRAM_MiB'),'latency_warmup_payloads_excluded':10,'latency_reference':protocol.get('latency_reference',{'kind':'rule','policy':'gmt_values'}),'elapsed_seconds':time.monotonic()-begin,'actual_mutated_state_online':True,'no_GT_actor_input':True,'official_TEST':False,'Full24':False,'TrackEval_raw_GT_duplicate_policy':'same permissive unmodified official TRAIN GT as preceding phases; duplicate counts in evaluation manifest','model_choice_or_hyperparameters_changed':False};save(done,result);save(out/'PROGRESS.json',{'status':'COMPLETE','frames':frames,'metrics':m});print('NATIVE_CLOSED_LOOP_COMPLETE',phase,name,video,m,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--case',required=True);p.add_argument('--phase',choices=['validation','heldout'],required=True);a=p.parse_args();run(a.video,a.case,a.phase)
