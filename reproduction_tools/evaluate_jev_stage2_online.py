"""True mutated-state full-video native decisions, atomic segmented recovery."""
import argparse,gzip,time,collections,traceback
import numpy as np
import torch
from jev_phase13_learning import *
from jev_phase13_runtime import *
from audit_jev_stage2_gta_free import phase13_prefix
from run_jev_phase10_closed_loop import raw_predictions,metrics

def load_policy(variant,seed,phase):
    if variant=='cosine':return None,None,1.
    p=OUT/f'{phase}_training_v1'/variant/f'seed{seed}'/'RESULT.json';r=json.loads(p.read_text());assert r['status']=='COMPLETE' and r['actual_updates']==r['target_updates'];c=r['checkpoint'];assert sha(c['path'])==c['SHA256'];ck=torch.load(c['path'],map_location='cpu');network=GlobalIdentityJev(variant).cuda().eval();network.load_state_dict(ck['model'],strict=True);return network,{'training_result':str(p),'training_result_SHA256':sha(p),'checkpoint':c,'training_binding':r['binding'],'dataset_SHA256':r['dataset_SHA256']},r['temperature']

def main(video,variant,seed=20261009,phase='formal',chunk_size=256,no_calibration=False):
    protect();source=binding();assert not source['dirty'];assert video in VAL
    protocol=json.loads((REPORTS/'TRAINING_PROTOCOL.json').read_text());assert json.loads((REPORTS/'NATIVE_PARITY.json').read_text())['status']=='PASS'
    policy,trained,temp=load_policy(variant,seed,phase);temp=1. if no_calibration else temp;name=f'{variant}_seed{seed}'+('_no_calibration' if no_calibration else '');out=OUT/f'{phase}_online_v1'/name/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():assert json.loads((out/'RESULT.json').read_text())['binding']==source;return
    torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;model=build_tracker(video,policy,variant=variant,temperature=(temp,1.),react_learned=False);values,frames,reader=cache_inputs(video)
    ann=json.loads(ANNOTATIONS.read_text());images={(i['frame_id']-1,i['view_id']-1):i for i in ann['images'] if i['video_id']==video}
    save(out/'START.json',{'binding':source,'trained':trained,'frames':frames,'variant':variant,'seed':seed,'temperature':temp,'strict_online':True,'actor_GT_labels':False,'REACT':'UNTRAINED_COSINE_FALLBACK','MEMORY':'UNTRAINED_NATIVE_WRITE'})
    counts=collections.Counter();parts=[];prefix=None;cursor=0;journal=[None];commits=[None];last_write=[0.];executor=model.jev_stage2_executor
    def question(**d):
        c=d['context'];batch=d['batch'];rows=c.get('rows',list(range(len(d['logits']))));journal[0].write(json.dumps({'key':[video,c['frame'],c['view']],'task':d['task'],'refs':d['refs'],'rows':rows,'logits':d['logits'].cpu().tolist(),'legal':batch['legal'][0].cpu().tolist()})+'\n')
    def after(**d):
        from gtr.modeling.meta_arch.gtr_rcnn import poss_ids,old_reids
        current=d['instances'][-1];ids=current.track_ids.cpu().tolist();assert len(ids)==len(set(ids));counts['payloads']+=1;counts['detections']+=len(ids)
        for e in d['events']:counts[e['action']]+=1;counts['native_memory_WRITEs']+=1
        commits[0].write(json.dumps({'key':[video,d['frame'],d['view']],'ids':ids,'id_count':d['id_count'],'events':d['events'],'hits':{str(t):int(d['hits'][t]) for t in ids},'Gallery_lengths':{str(t):len(d['galleries'][t]) for t in ids},'possible_ids':sorted(poss_ids.poss_ids),'old_reids_IDs':old_reids.old_reids[0].track_ids.cpu().tolist() if old_reids.old_reids else [],'trajectory_RNG_draws':len(model._jev_trajectory_rng._trajectory_draws)})+'\n')
        if time.monotonic()-last_write[0]>=15:save(out/'PROGRESS.json',{'status':'RUNNING','key':[video,d['frame'],d['view']],'frames':frames,'counts':dict(counts),'source_commit':source['source_commit']});last_write[0]=time.monotonic()
    executor.observer=question;executor.commit_observer=after;resume=out/'RUN_RESUME.json';elapsed_prior=0.
    if resume.exists():
        r=json.loads(resume.read_text());assert r['binding']==source and r['trained']==trained;assert sha(r['prefix']['path'])==r['prefix']['SHA256'];prefix=r['prefix']['path'];cursor=r['next_frame'];parts=r['parts'];counts.update(r['counts']);executor.latency=r['latencies'];elapsed_prior=r['seconds']
    class SegmentBoundary(Exception):pass
    begin=time.monotonic()
    while cursor<frames:
        stop=min(frames,cursor+chunk_size);attempt=0
        while (out/f'chunk{cursor:06d}_attempt{attempt:03d}').exists():attempt+=1
        part=out/f'chunk{cursor:06d}_attempt{attempt:03d}';part.mkdir();journal[0]=gzip.open(part/'QUESTIONS.jsonl.gz','wt');commits[0]=gzip.open(part/'COMMITS.jsonl.gz','wt');next_prefix=[None]
        def before(**d):
            if (d['frame'],d['view'])==(stop,0):
                p=out/f'UNFROZEN_RECOVERY_SLOT_{(stop//chunk_size)%2}.pth';atomic_torch(p,phase13_prefix(model,d));next_prefix[0]={'path':str(p),'SHA256':sha(p)};raise SegmentBoundary()
        model.jev_native_prefix_observer=before
        try:
            with torch.no_grad():raw,_=run(model,values,frames,prefix=torch.load(prefix,map_location='cuda:0') if prefix else None)
            finished=True
        except SegmentBoundary:finished=False
        finally:journal[0].close();commits[0].close()
        item={'start':cursor,'stop_exclusive':stop,'journal':str(part/'QUESTIONS.jsonl.gz'),'journal_SHA256':sha(part/'QUESTIONS.jsonl.gz'),'commits':str(part/'COMMITS.jsonl.gz'),'commits_SHA256':sha(part/'COMMITS.jsonl.gz')};parts.append(item);save(part/'COMPLETE.json',{'status':'COMPLETE','binding':source,'part':item,'total_committed_payloads':counts['payloads']})
        if finished:assert stop==frames;break
        assert next_prefix[0];save(resume,{'status':'RESUMABLE','binding':source,'trained':trained,'next_frame':stop,'prefix':next_prefix[0],'parts':parts,'counts':dict(counts),'latencies':executor.latency,'seconds':elapsed_prior+time.monotonic()-begin});prefix=next_prefix[0]['path'];cursor=stop;print('PHASE13_ONLINE_SEGMENT',name,video,cursor,frames,flush=True)
    assert counts['payloads']==2*frames-1
    for field,filename in [('journal','QUESTIONS.jsonl.gz'),('commits','COMMITS.jsonl.gz')]:
        with (out/filename).open('wb') as h:
            for part in parts:assert sha(part[field])==part[field+'_SHA256'];h.write(Path(part[field]).read_bytes())
    # Predictions are frozen before any evaluator/GT-label audit is invoked.
    predictions=raw_predictions(raw,images);save(out/'RAW_PREDICTIONS.json',predictions);strict,evalpath=metrics(out/'RAW_PREDICTIONS.json',[video],out/'strict_eval')
    from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
    filtered=model._remove_short_track(raw) if model.min_track_len>0 else raw
    if model.roi_heads.delay_cls:filtered=model._delay_cls(filtered,video_id=video)
    order=[values[v*frames+f] for f in range(frames) for v in range(2)];processed=CustomRCNN._postprocess(filtered,order,[(0,0)]*len(order),not_clamp_box=model.not_clamp_box);canonical=[]
    for index,o in enumerate(processed):
        f,v=divmod(index,2);i=o['instances']
        for box,ref,score in zip(i.pred_boxes.tensor.cpu().tolist(),i.track_ids.cpu().tolist(),i.scores.cpu().tolist()):
            x,y,x2,y2=box;canonical.append({'image_id':images[f,v]['id'],'track_id':int(ref),'bbox':[x,y,max(0.,x2-x),max(0.,y2-y)],'score':float(score),'category_id':1})
    save(out/'CANONICAL_PREDICTIONS.json',canonical);canonical_metrics,canpath=metrics(out/'CANONICAL_PREDICTIONS.json',[video],out/'canonical_eval')
    from jev_phase7_offline import IdentityEvaluator
    evaluator=IdentityEvaluator(video);aligned=evaluator.align(predictions);identity=evaluator.summarize(aligned);save(out/'IDENTITY_AUDIT.json',identity)
    from jev_phase13_online_audit import audit
    decisions=audit(video,predictions,out/'QUESTIONS.jsonl.gz',out/'COMMITS.jsonl.gz',reader);save(out/'DECISION_AUDIT.json',decisions);save(out/'LATENCY_SAMPLES.json',executor.latency)
    def quantile(key):
        a=[r[key] for r in executor.latency[10:]];return dict(zip(['p50','p95','max'],map(float,np.quantile(a,[.5,.95,1])))) if a else None
    result={'status':'COMPLETE','binding':source,'variant':variant,'seed':seed,'phase':phase,'trained':trained,'frames':frames,'counts':dict(counts),'temperature':temp,'strict_online_metrics':strict,'canonical_future_filtered_metrics':canonical_metrics,'strict_evaluator':{'path':str(evalpath/'metrics.json'),'SHA256':sha(evalpath/'metrics.json')},'canonical_evaluator':{'path':str(canpath/'metrics.json'),'SHA256':sha(canpath/'metrics.json')},'raw_predictions':{'path':str(out/'RAW_PREDICTIONS.json'),'SHA256':sha(out/'RAW_PREDICTIONS.json')},'canonical_predictions':{'path':str(out/'CANONICAL_PREDICTIONS.json'),'SHA256':sha(out/'CANONICAL_PREDICTIONS.json')},'commits_SHA256':sha(out/'COMMITS.jsonl.gz'),'questions_SHA256':sha(out/'QUESTIONS.jsonl.gz'),'identity_summary':{k:v for k,v in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},'cross_camera':{k:v for k,v in identity['identity_index_query'].items() if k!='queries'},'decision_summary':{k:v for k,v in decisions.items() if k!='action_outcomes'},'stage2_latency_with_audit_instrumentation_ms':quantile('total_stage2_ms'),'cached_stage2_only_wall_seconds_including_eval':elapsed_prior+time.monotonic()-begin,'full_FPS':None,'efficiency_scope':'cache-backed validation is not full FPS; isolated live benchmark disables journals/evaluator','actual_mutated_state_online':True,'GTA_throw_mock':True,'same_camera_ID_capacity_one':True,'bootstrap_payloads':1,'heldout':'SEALED','Full24':False,'official_TEST':False}
    save(out/'RESULT.json',result);save(out/'PROGRESS.json',{'status':'COMPLETE','strict_metrics':strict,'frames':frames})
    # These two mutable recovery slots are not frozen scientific artifacts;
    # completed part SHA and actual commits/predictions remain reproducible.
    for p in out.glob('UNFROZEN_RECOVERY_SLOT_*.pth'):p.unlink()
    print('PHASE13_FULL_ONLINE_COMPLETE',name,video,strict,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--phase',choices=['formal','onpolicy'],default='formal');p.add_argument('--chunk-size',type=int,default=256);p.add_argument('--no-calibration',action='store_true');a=p.parse_args()
    try:main(a.video,a.variant,a.seed,a.phase,a.chunk_size,a.no_calibration)
    except Exception:save(OUT/'online_failures'/f'{a.variant}_seed{a.seed}_video{a.video}_{time.time_ns()}.json',{'status':'FAIL','binding':binding(),'traceback':traceback.format_exc()});raise
