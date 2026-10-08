"""Re-execute frozen causal branches in the actual mutable production loop."""
import argparse,json,time
from pathlib import Path
import torch
from jev_phase10_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,prefix_state,fingerprint
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy

def predictions(raw,evaluator,start_key,end_frame):
 images={(int(i['frame_id'])-1,int(i['view_id'])-1):i for i in evaluator.images.values()};output=[]
 for pos,inst in enumerate(raw):
  frame,view=divmod(pos,2)
  if (frame,view)<tuple(start_key[1:])or frame>end_frame:continue
  image=images[(frame,view)];h,w=inst.image_size
  for row,(box,track,score)in enumerate(zip(inst.pred_boxes.tensor.detach().cpu().tolist(),inst.track_ids.detach().cpu().tolist(),inst.scores.detach().cpu().tolist())):
   x1,y1,x2,y2=box;sx=image['width']/w;sy=image['height']/h
   output.append({'image_id':image['id'],'track_id':int(track),'bbox':[x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy],'score':float(score),'category_id':1})
 return output

def run(video,start,stop,diagnostic):
 protect();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;source=binding()
 if not diagnostic:
  assert not source['dirty'];gate=json.loads((REPORTS/'NATIVE_STATE_PARITY.json').read_text());assert gate['status']=='PASS'and gate['fixed_original_records']==221
 plan=json.loads((REPORTS/'PHASE10_NATIVE_FORK_PLAN.json').read_text());chosen=[e for e in plan['events']if e['key'][0]==video][start:stop]
 if diagnostic:chosen=chosen[:1]
 cap=OUT/'native_capture_v1'/f'video{video:02d}'/'compat';capture=json.loads((cap/'RESULT.json').read_text());registry={tuple(r['key']):r for r in capture['prefixes']};factual={tuple(r['key']):r for r in json.loads((cap/'FULL_NATIVE_TRACE.json').read_text())}
 output=OUT/('native_fork_smoke_'+os.environ.get('JEV_PHASE10_SMOKE_VERSION','v1')if diagnostic else'native_forks_v3');output.mkdir(exist_ok=True)
 from jev_phase7_offline import IdentityEvaluator
 from jev_phase8_utility import effects
 evaluator=IdentityEvaluator(video);model=build_model(video);allinputs=inputs(video,capture['frames']);results=[]
 for event in chosen:
  key=tuple(event['key']);row=event['row'];prefix=registry[key];assert sha(prefix['path'])==prefix['sha256'];old=event['original_snapshot'];assert sha(old['path'])==old['sha256'];snapshot=torch.load(old['path'],map_location='cpu');anchors=snapshot['offline_prefix_identity'];end=min(key[1]+(1 if diagnostic else 31),capture['frames']-1);branches={};traces={}
  for tag,spec in event['branch_specs'].items():
   out=output/f'event{event["index"]:03d}'/tag;out.mkdir(parents=True,exist_ok=True);complete=out/'RESULT.json'
   if complete.exists():
    value=json.loads(complete.read_text());assert value['binding']==source and value['spec_SHA256']==fingerprint(spec);branches[tag]=value;traces[tag]=json.loads((out/'NATIVE_TRACE.json').read_text());continue
   begin=time.monotonic();save(out/'START.json',{'binding':source,'spec':spec,'prefix':prefix,'end_frame':end,'diagnostic_only':diagnostic})
   recorder=NativeProductionPrefixRecorder(model,[],out);policy=CandidateValuePolicy('gmt_compat');model.jev_native_match_override={k:spec[k]for k in ['key','row','tag','candidate_ids','pairs']};model.__dict__.pop('_jev_native_intervention_last',None)
   before=[False];current=[None]
   def observe_before(**values):
    here=(video,values['frame'],values['view'])
    model.jev_candidate_policy=policy if here==key else None
    model._jev_candidate_new_rows=();model.__dict__.pop('_jev_candidate_last',None)
    if here==key:
     assert fingerprint(prefix_state(model,**values))==fingerprint(torch.load(prefix['path'],map_location=model.device)),'fork initial prefix differs'
     before[0]=True
    recorder.before(**values)
   def after(**values):
    if (video,values['frame'],values['view'])==key:current[0]=dict(model._jev_native_intervention_last)
    recorder.after(**values)
    for r in recorder.trace[-1]['events']:
     if r['track']<0:raise AssertionError('uncommitted ID')
    ids=recorder.trace[-1]['ids'];assert len(set(ids))==len(ids),'per-camera duplicate ID'
   model.jev_native_prefix_observer=observe_before;model.jev_candidate_commit_observer=after
   with torch.no_grad():raw,_=NativeStateForkAdapter(model).run(prefix['path'],allinputs,stop_frame=end)
   assert before[0]and current[0]is not None
   if tag in ['CONTROL','KEEP_FACTUAL','KEEP_ACCEPT']:
    assert all(t==factual[tuple(t['key'])]for t in recorder.trace),'CONTROL/KEEP entire native state, event or RNG deviates from factual future'
   save(out/'NATIVE_TRACE.json',recorder.trace);traces[tag]=recorder.trace
   if diagnostic:
    value={'status':'PASS','diagnostic_only':True,'binding':source,'spec_SHA256':fingerprint(spec),'native':current[0],'all_camera_payload_IDs_unique':True,'end_frame':end,'elapsed_seconds':time.monotonic()-begin}
   else:
    pred=predictions(raw,evaluator,key,end);save(out/'PREDICTIONS.json',pred);aligned=evaluator.align(pred);immediate=aligned[(key[1],key[2],row)];mapping={int(t):int(g)for t,g in anchors['reliable'].items()};gt=event['offline_GT'];actual=immediate['id'];wanted=event['correct_candidate_ids'];native_id=current[0]['native_existing_ids'][row]
    value={'status':'PASS','binding':source,'spec_SHA256':fingerprint(spec),'prefix_SHA256':prefix['sha256'],'native':current[0],'all_camera_payload_IDs_unique':True,'actual_committed_id':actual,'current_candidate_origin_qualified':native_id>=0 and native_id==actual,'desired_candidate_committed':actual in wanted,'immediate_anchored_correct':None if gt is None or actual not in mapping else mapping[actual]==gt,'H32_complete':end==key[1]+31,'horizons':{str(h):effects(aligned,anchors,key[1],h,{gt}if gt is not None else set())for h in [8,16,32]},'prediction_SHA256':sha(out/'PREDICTIONS.json'),'native_trace_SHA256':sha(out/'NATIVE_TRACE.json'),'elapsed_seconds':time.monotonic()-begin,'mutated_future_recomputed':True,'no_GT_in_actor':True,'source':'actual resumed sliding_inference_GMT, native perception and lifecycle'}
   branches[tag]=value;save(complete,value);print('NATIVE_BRANCH_COMPLETE',video,event['index'],tag,round(value['elapsed_seconds'],2),flush=True)
  control=branches['CONTROL'];keep=next(t for t in branches if t.startswith('KEEP'));assert traces['CONTROL']==traces[keep],'explicit KEEP and CONTROL differ'
  verified=[];changed=[]
  if not diagnostic:
   for tag,b in branches.items():
    old=json.loads(Path(event['branch_specs'][tag]['old_effect_path']).read_text())
    for h in [8,16,32]:
     e=b['horizons'][str(h)];c=control['horizons'][str(h)];e['delta_utility']=e['utility']-c['utility'];e['delta_utility_birth_zero']=e['utility_birth_zero']-c['utility_birth_zero'];e['delta_target_wrong']=e['target_counts'].get('wrong',0)-c['target_counts'].get('wrong',0);e['delta_wrong_duration_camera_frames']=e['wrong_identity_duration_camera_frames']-c['wrong_identity_duration_camera_frames']
    comparable=['counts','utility','utility_birth_zero','wrong_identity_duration_camera_frames']
    differs=any(b['horizons'][str(h)].get(f)!=old['horizons'][str(h)].get(f)for h in [8,16,32]for f in comparable)
    if differs:changed.append(tag)
    if event['proposal_correct']is False and control['immediate_anchored_correct']is False and b['current_candidate_origin_qualified']and b['desired_candidate_committed']and b['immediate_anchored_correct']is True and b['H32_complete']and b['horizons']['32']['delta_utility']>0 and b['horizons']['32']['delta_utility_birth_zero']>0:verified.append(tag)
  result={'status':'PASS','binding':source,'key':list(key),'row':row,'group':event['group'],'sampling_records':event['sampling_records'],'distribution':event['distribution'],'index':event['index'],'branches':branches,'CONTROL_KEEP_complete_state_PASS':True,'CONTROL_full_factual_state_events_RNG_PASS':True,'verified_candidate_corrective_branches':verified,'changed_old_horizon_branches':changed,'diagnostic_only':diagnostic};save(output/f'event{event["index"]:03d}'/'EVENT_RESULT.json',result);results.append(result)
  save(output/f'PROGRESS_video{video:02d}_{start:03d}_{stop or 999:03d}.json',{'status':'RUNNING','completed_events':len(results),'total_events':len(chosen),'last_index':event['index'],'diagnostic_only':diagnostic})
 save(output/f'PROGRESS_video{video:02d}_{start:03d}_{stop or 999:03d}.json',{'status':'COMPLETE','completed_events':len(results),'total_events':len(chosen),'diagnostic_only':diagnostic})
 print('NATIVE_FORK_SHARD_COMPLETE',video,start,stop,len(results),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);p.add_argument('--diagnostic',action='store_true');a=p.parse_args();run(a.video,a.start,a.stop,a.diagnostic)
