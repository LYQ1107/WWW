"""Complete live-GMT cold native controls; GT only observes and evaluates."""
import argparse,collections,gzip,torch
from gmt_jev_runtime import *
from gmt_jev_opportunity import OfflineCertificates,native_assignment
from gmt_jev_evaluate import evaluate,duration
def main(kind):
 from gtr.modeling.gmt_jev import IdentityCorrection
 from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
 from run_jev_phase10_closed_loop import raw_predictions
 learned=kind.endswith('_learned');head_kind=kind.replace('_learned','')
 assert kind in ['threshold','rule','confidence','mlp_learned','attention_learned','cic_learned']
 assert read(REPORTS/'BASELINE_PARITY.json')['status']=='PASS_EXACT'
 folder=OUT/'online'/kind;folder.mkdir(parents=True,exist_ok=False)
 values,frames,images=metadata(16);seed();m,cfg,_=model();checkpoint=None
 if learned:
  from gmt_jev_learned_checkpoint import load_actor
  actor,checkpoint=load_actor(head_kind)
 else:actor=IdentityCorrection(kind).cuda().eval()
 m.gmt_identity_correction=actor;seed()
 labeler=OfflineCertificates(16,images);counts=collections.Counter();pending={};boundary={};boot=[False];last=[0.];begin=time.monotonic()
 def before(**d):
  boundary['id_count']=d['id_count']
  if d['first'] and not boot[0]:
   other=1-d['view'];i=d['instances'][other];labeler.update(i,0,other);actor.committed(i,other,0);boot[0]=True
 def observed(packet,current,galleries):
  f,v=packet['frame'],packet['view'];refs=packet['ids'].tolist();gt=labeler.labels(current,f,v);target,categories=labeler.targets(gt,refs)
  base=native_assignment(packet['base_scores'],packet['lengths'],m.overlap_thresh)
  changed=native_assignment(packet['corrected_scores'],packet['lengths'],m.overlap_thresh)
  pending[f,v]=dict(refs=refs,target=target,gt=gt,owners={t:labeler.clean(t) for t in refs},base=base,changed=changed,id_count=boundary['id_count'])
  counts.update(categories);counts['MATCH_rows']+=len(gt)
  counts['intervened_edges']+=int((packet['residual']!=0).sum());counts['eligible_edges']+=int(packet['eligible'].sum())
  for risk in ['low','medium','high']:counts['risk_'+risk+'_changed_rows']+=0
  mask=packet['residual'].ne(0).any(-1)
  for row in torch.where(mask)[0].tolist():
   margin=float(packet['features'][row,0,3]) if len(refs) else 0
   risk='high' if margin<.02 else 'medium' if margin<.05 else 'low';counts['risk_'+risk+'_changed_rows']+=1
 actor.observer=observed;m.jev_native_prefix_observer=before
 def after(**d):
  f,v=d['frame'],d['view'];inst=d['instances'][-1];ids=inst.track_ids.tolist();assert len(ids)==len(set(ids))
  s=pending.pop((f,v));counts['commits']+=1;counts['native_births']+=sum(t>s['id_count'] for t in ids)
  counts['native_Bank_restore_rows']+=sum(t<=s['id_count'] and t not in s['refs'] for t in ids)
  for row,(actual,y,g,b,c) in enumerate(zip(ids,s['target'],s['gt'],s['base'],s['changed'])):
   if y>=0:counts['unique_certified_correct_active_rows']+=1;counts['unique_certified_correct_preserved']+=actual==s['refs'][y]
   if b==c:continue
   counts['changed_active_assignment_rows']+=1
   if y<0 or b<0:counts['UNKNOWN_changed_base_Bank_or_target']+=1;continue
   target=s['refs'][y];base_id=s['refs'][b]
   if s['owners'].get(base_id) is None:counts['UNKNOWN_changed_base_owner']+=1;continue
   if base_id!=target and s['owners'][base_id]!=g and actual==target:counts['N01']+=1
   elif base_id==target and actual!=target and (actual>s['id_count'] or (s['owners'].get(actual) is not None and s['owners'][actual]!=g)):counts['N10']+=1
   elif actual!=base_id:counts['UNKNOWN_changed_actual_owner']+=1
  labeler.update(inst,f,v);actor.committed(inst,v,f)
  if time.monotonic()-last[0]>20:
   save(folder/'PROGRESS.json',dict(status='FULL_NATIVE_GMT_CLOSED_LOOP',frame=f,frames=frames,counts=dict(counts)));print('GMT_ONLINE_PROGRESS',kind,f,frames,dict(counts),flush=True);last[0]=time.monotonic()
 m.jev_candidate_commit_observer=after
 with torch.no_grad():raw,_=run(m,values,frames)
 predictions=raw_predictions(raw,images);path=folder/'RAW_PREDICTIONS.json.gz'
 with gzip.open(path,'wt') as f:json.dump(predictions,f)
 if learned:
  from gmt_jev_train_matched_quality import geometry_sha
  reference=read(REPORTS/'ORIGINAL_GMT_METRICS.json')['predictions'];assert sha(reference['path'])==reference['SHA256']
  with gzip.open(reference['path'],'rt') as f:original_predictions=json.load(f)
  assert geometry_sha(predictions)==geometry_sha(original_predictions),'Learned correction changed frontend predictions'
 # Exact original stock end filtering and postprocessing, separate from raw.
 stock=m._remove_short_track(list(raw)) if m.min_track_len>0 else list(raw)
 if m.roi_heads.delay_cls:stock=m._delay_cls(stock,video_id=16)
 batch=[values[v*frames+f] for f in range(frames) for v in range(2)]
 stock=CustomRCNN._postprocess(stock,batch,[(0,0)]*len(batch),not_clamp_box=m.not_clamp_box)
 stock_pred=raw_predictions([r['instances'] for r in stock],images);stockpath=folder/'STOCK_PREDICTIONS.json.gz'
 with gzip.open(stockpath,'wt') as f:json.dump(stock_pred,f)
 parameters=dict(full_system_instantiated=sum(p.numel() for p in m.parameters()),extra_trainable=sum(p.numel() for p in actor.parameters()),deployed=sum(p.numel() for p in m.parameters()))
 del m,raw,stock;torch.cuda.empty_cache()
 raw_metric=evaluate(predictions,folder/'raw_metrics');stock_metric=evaluate(stock_pred,folder/'stock_metrics')
 result=dict(status='COMPLETE',kind=kind,frames=frames,video=16,primary_raw=raw_metric,secondary_stock=stock_metric,raw_predictions=ref(path),stock_predictions=ref(stockpath),counts=dict(counts),identity=duration(predictions),
  native_mutated_state=True,GT_actor_inputs=False,no_new_gradient_updates=True,parameters=parameters,full_image_FPS=None,independent_validation=False,
  row_certificate_scope='Same-current-state raw GMT accepted MATCH reference versus corrected current native commit, with unique first-past targets. Rejected baseline has UNKNOWN original Bank outcome. Not a whole-plan causal or cross-run numeric-ID claim.',
  seconds=time.monotonic()-begin,source=binding())
 if learned:
  result.update(trained_checkpoint=checkpoint,risk_mode='learned',training_steps=300,
   raw_geometry_and_scores_SHA256=geometry_sha(predictions),raw_frontend_prediction_parity=True,
   effective_config_SHA256=hashlib.sha256(cfg.dump().encode()).hexdigest(),annotations=ref(ANNOTATIONS),
   selection='All three fixed step300 heads; no selection by development quality')
 save(REPORTS/f'ONLINE_{kind}.json',result);print('GMT_ONLINE_COMPLETE',kind,raw_metric['metrics'],raw_metric['CVIDF1'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['threshold','rule','confidence','mlp_learned','attention_learned','cic_learned'],required=True);main(p.parse_args().kind)
