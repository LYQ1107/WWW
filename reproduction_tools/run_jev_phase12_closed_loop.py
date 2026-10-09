"""Segmented native mutated-state actors; evaluate GT only after saved inference."""
import argparse,gzip,time,collections,traceback
import numpy as np
import torch
from jev_phase12_common import *
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_candidate_assignment import assign_candidate_values
from gtr.modeling.jev_native_state import NativeStateForkAdapter,prefix_state
from gtr.modeling.visual_jev_mcmot import VisualJev
from gtr.modeling.visual_jev_mcmot.baselines import LegacyNumericAdapter,NumericalOnlyVisualAdapter
from gtr.modeling.visual_jev_mcmot.lifecycle_controller import VisualLifecycleController
from run_jev_phase10_closed_loop import metadata,raw_predictions,metrics

def configure(model,case):
    if case['kind']=='model':
        c=case['checkpoint'];assert sha(c['path'])==c['SHA256'];saved=torch.load(c['path'],map_location='cuda:0')
        name=case['model'];network=LegacyNumericAdapter(name) if name.startswith('Candidate') else NumericalOnlyVisualAdapter() if name=='numerical_only' else VisualJev(name)
        network.cuda().eval();network.load_state_dict(saved['model'])
        model.visual_jev_controller=VisualLifecycleController(network,'MATCH_ONLY',supervision='joint',risk=case['risk'],temperature=case['temperature'],qualified_tasks=('MATCH',))
        model.visual_jev_enabled=True;model.jev_candidate_policy=CandidateValuePolicy('gmt_compat')
    else:
        model.visual_jev_enabled=False;model.jev_candidate_policy=None if case['kind']=='off' else CandidateValuePolicy('gmt_values')

def run(video,name,smoke_frames=0,chunk_size=256):
    protect();source=binding();assert not source['dirty'];assert video in VAL
    protocol=json.loads((REPORTS/'ONLINE_PROTOCOL.json').read_text());assert protocol['status']=='FROZEN_BEFORE_MOT_VALIDATION'
    for file,digest in protocol['scripts_SHA256'].items():assert sha(ROOT/file)==digest,file
    case=next(c for c in protocol['cases'] if c['name']==name)
    root='online_smoke_v1' if smoke_frames else 'validation_closed_loop_v1'
    out=OUT/root/name/f'video{video:02d}'/f'chunk{chunk_size}' if smoke_frames else OUT/root/name/f'video{video:02d}'
    out.mkdir(parents=True,exist_ok=True);done=out/'RESULT.json'
    if done.exists():assert json.loads(done.read_text())['binding']==source;print('ONLINE_ALREADY_COMPLETE',name,video);return
    torch.set_num_threads(1);torch.manual_seed(20261008);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=build_model(video);configure(model,case);rows,images,frames=metadata(video)
    if smoke_frames:
        assert 0<smoke_frames<=160;original=frames;frames=smoke_frames;rows=[rows[v*original+f] for v in range(2) for f in range(frames)]
    save(out/'START.json',{'binding':source,'case':case,'frames':frames,'strict_online':True,'chunk_size':chunk_size,'actor_GT_fields':False})
    counts=collections.Counter();latencies=[];native=[];stages=[];parts=[];start=0;prefix=None;state=[None];journal=[None];commits=[None];lastref=[0.];stamp=[0.]
    reference=CandidateValuePolicy('gmt_compat')
    def before(**d):
        lo=max(0,d['frame']+1-model.test_len)*2;hi=d['frame']*2+d['view'];active=[i.track_ids for i in d['instances'][lo:hi] if i.has('track_ids')]
        state[0]={'id_count':int(d['id_count']),'gallery_lengths':{int(t):len(g) for t,g in d['galleries'].items()},'active':set(torch.cat(active).cpu().tolist()) if active else set(),'begin':time.perf_counter()}
        model.__dict__.pop('_jev_candidate_last',None)
        if (d['frame'],d['view'])==(128,0):
            path=out/'CAUSAL_PREFIX_128_0.pth'
            if not path.exists():torch.save(prefix_state(model,**d),path)
    def profile(**d):
        batch=d.pop('batch');torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        v,n=reference.score(batch);assign_candidate_values(v,n,batch.candidate_ids,batch.legal_mask,mode=reference.assignment_mode,legacy_thresholds=batch.thresholds)
        torch.cuda.synchronize();ref=(time.perf_counter()-begin)*1000;lastref[0]=ref
        d.update(reference_policy_assignment_ms=ref,extra_policy_assignment_ms=d['policy_assignment_ms']-ref,peak_VRAM_MiB=torch.cuda.max_memory_allocated()/1048576,reference_temporary_bytes=torch.cuda.max_memory_allocated()-allocated)
        latencies.append(d)
    def after(**d):
        old=state[0];instance=d['instances'][-1];ids=instance.track_ids.cpu().tolist();assert len(set(ids))==len(ids),'same-camera duplicate IDs'
        counts['payloads']+=1;counts['detections']+=len(ids);events=[]
        for row,t in enumerate(ids):
            event={'birth':t>old['id_count'],'write':len(d['galleries'][t])>old['gallery_lengths'].get(t,0),'bank_reactivation':t<=old['id_count'] and t not in old['active']};events.append(event)
            counts['births']+=event['birth'];counts['memory_writes']+=event['write'];counts['bank_reactivations']+=event['bank_reactivation'];counts['memory_skips']+=not event['write']
        key=[video,int(d['frame']),int(d['view'])]
        commit={'key':key,'ids':ids,'id_count':int(d['id_count']),'hits':{str(t):int(d['hits'][t]) for t in ids},'gallery_lengths':{str(t):len(d['galleries'][t]) for t in ids},'events':events}
        commits[0].write(json.dumps(commit)+'\n');packet=d['candidate']
        if packet is not None:
            b=packet['batch'];proposal=[]
            for row in range(len(ids)):
                cols=torch.where(b.evidence12[row,:,2]>0)[0];proposal.append(b.candidate_ids[int(cols[0])] if len(cols) else None)
            journal[0].write(json.dumps({'key':key,'candidate_ids':b.candidate_ids,'initial_proposal_ids':proposal,'GMT_scores':b.scores.cpu().tolist(),'thresholds':b.thresholds.cpu().tolist(),'candidate_values':packet['candidate_values'].cpu().tolist(),'legal_mask':b.legal_mask.cpu().tolist(),'existing_ids':packet['assignment'].existing_ids,'NEW_rows':packet['assignment'].new_rows,'committed_ids':ids})+'\n')
            counts['MATCH_DEFER_rows']+=len(packet['assignment'].new_rows)
        if model.visual_jev_enabled:
            controller=model.visual_jev_controller
            for record in controller.records:
                if record['task']=='MATCH':
                    counts['risk_abstain_rows']+=record['abstain_rows'];counts['fallback_rows']+=len(record['fallback_rows']);counts['match_questions']+=record['questions'];counts['shared_state_encodes']+=record['memory_reused_for_questions']>0
            stages.extend(controller.timings);controller.records.clear();controller.timings.clear()
        native.append({'key':key,'native_association_commit_ms_excluding_paired_reference':(time.perf_counter()-old['begin'])*1000-(lastref[0] if packet else 0)})
        if time.monotonic()-stamp[0]>15:save(out/'PROGRESS.json',{'status':'RUNNING','last_key':key,'frames':frames,'counts':dict(counts),'source_commit':source['source_commit']});stamp[0]=time.monotonic()
    model.jev_candidate_commit_observer=after
    if model.jev_candidate_policy is not None:model.jev_candidate_latency_observer=profile
    resume=out/'RUN_RESUME.json'
    if resume.exists():
        saved=json.loads(resume.read_text());assert saved['binding']==source and saved['case']==case;prefix=saved['prefix']['path'];assert sha(prefix)==saved['prefix']['SHA256'];start=saved['next_frame'];parts=saved['parts'];counts.update(saved['counts']);latencies=saved['latencies'];native=saved['native'];stages=saved['stages']
    class SegmentBoundary(Exception):pass
    begin=time.monotonic()
    while start<frames:
        stop=min(frames,start+chunk_size);attempt=0
        while (out/f'chunk{start:06d}_attempt{attempt:03d}').exists():attempt+=1
        part=out/f'chunk{start:06d}_attempt{attempt:03d}';part.mkdir();journal[0]=gzip.open(part/'CANDIDATE_JOURNAL.jsonl.gz','wt');commits[0]=gzip.open(part/'NATIVE_COMMITS.jsonl.gz','wt');nextprefix=[None]
        def segmented(**d):
            if (d['frame'],d['view'])==(stop,0):
                path=out/f'UNFROZEN_RECOVERY_SLOT_{(stop//chunk_size)%2}.pth';tmp=path.with_suffix('.tmp');torch.save(prefix_state(model,**d),tmp);tmp.replace(path);nextprefix[0]={'path':str(path),'SHA256':sha(path)};raise SegmentBoundary()
            before(**d)
        model.jev_native_prefix_observer=segmented
        try:
            with torch.no_grad():
                if prefix is None:instances,_=model.sliding_inference_GMT(rows,2,[None,None,list(range(len(rows)))],native_raw=True)
                else:instances,_=NativeStateForkAdapter(model).run(prefix,rows,stop_frame=frames-1)
            finished=True
        except SegmentBoundary:finished=False
        finally:journal[0].close();commits[0].close()
        parts.append({'start_frame':start,'stop_exclusive':stop,'journal':str(part/'CANDIDATE_JOURNAL.jsonl.gz'),'journal_SHA256':sha(part/'CANDIDATE_JOURNAL.jsonl.gz'),'commits':str(part/'NATIVE_COMMITS.jsonl.gz'),'commits_SHA256':sha(part/'NATIVE_COMMITS.jsonl.gz')})
        save(part/'COMPLETE.json',{'status':'COMPLETE','binding':source,'part':parts[-1],'committed_payloads':counts['payloads']})
        if finished:assert stop==frames;break
        assert nextprefix[0];save(resume,{'binding':source,'case':case,'next_frame':stop,'prefix':nextprefix[0],'parts':parts,'counts':dict(counts),'latencies':latencies,'native':native,'stages':stages});prefix=nextprefix[0]['path'];start=stop
        print('NATIVE_SEGMENT_COMMITTED',name,video,start,frames,flush=True)
    for key,filename in [('journal','CANDIDATE_JOURNAL.jsonl.gz'),('commits','NATIVE_COMMITS.jsonl.gz')]:
        with (out/filename).open('wb') as handle:
            for part in parts:assert sha(part[key])==part[key+'_SHA256'];handle.write(Path(part[key]).read_bytes())
    # First dominant-camera bootstrap allocates IDs before the association
    # observer exists. Count its real predictions separately, never invent a
    # synthetic candidate commit. Subsequent payloads must be exactly complete.
    assert counts['payloads']==frames*2-1,(counts['payloads'],frames)
    raw=raw_predictions(instances,images);save(out/'RAW_PREDICTIONS.json',raw)
    counts['bootstrap_payloads']=1;counts['bootstrap_detections']=len(instances[0])
    if smoke_frames:
        save(done,{'status':'PASS','binding':source,'frames':frames,'counts':dict(counts),'raw_predictions_SHA256':sha(out/'RAW_PREDICTIONS.json'),'commits_SHA256':sha(out/'NATIVE_COMMITS.jsonl.gz'),'strict_online':True});return
    # Canonical GMT uses complete-video minimum-length filtering. Keep it as a
    # secondary benchmark and evaluate the saved causal committed stream too.
    from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
    filtered=model._remove_short_track(instances) if model.min_track_len>0 else instances
    if model.roi_heads.delay_cls:filtered=model._delay_cls(filtered,video_id=video)
    batchinputs=[rows[v*frames+f] for f in range(frames) for v in range(2)]
    processed=CustomRCNN._postprocess(filtered,batchinputs,[(0,0)]*len(batchinputs),not_clamp_box=model.not_clamp_box);pred=[]
    for index,value in enumerate(processed):
        f,v=divmod(index,2);im=images[f,v];i=value['instances']
        for box,t,score in zip(i.pred_boxes.tensor.cpu().tolist(),i.track_ids.cpu().tolist(),i.scores.cpu().tolist()):
            x,y,x2,y2=box;pred.append({'image_id':im['id'],'track_id':int(t),'bbox':[x,y,max(0.,x2-x),max(0.,y2-y)],'score':float(score),'category_id':1})
    save(out/'PREDICTIONS.json',pred);canonical,_=metrics(out/'PREDICTIONS.json',[video],out/'canonical_eval');strict,_=metrics(out/'RAW_PREDICTIONS.json',[video],out/'strict_eval')
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase10_online_audit import audit
    evaluator=IdentityEvaluator(video);aligned=evaluator.align(raw);identity=evaluator.summarize(aligned);save(out/'IDENTITY_METRICS.json',identity)
    diagnostic=audit(aligned,out/'CANDIDATE_JOURNAL.jsonl.gz',out/'NATIVE_COMMITS.jsonl.gz');save(out/'ONLINE_IDENTITY_AUDIT.json',diagnostic)
    with gzip.open(out/'NATIVE_COMMITS.jsonl.gz','rt') as handle:stream=list(map(json.loads,handle))
    from jev_phase12_paired_online import paired_h32
    causal=paired_h32(model,case,rows,images,evaluator,aligned,stream,out/'CAUSAL_PREFIX_128_0.pth',out)
    save(out/'LATENCY_SAMPLES.json',{'policy':latencies,'native_steps':native,'typed_MATCH':stages})
    def quantile(data,key):
        vals=[r[key] for r in data[10:]];return dict(zip(['p50','p95','max'],map(float,np.quantile(vals,[.5,.95,1])))) if vals else None
    result={'status':'COMPLETE','binding':source,'case':case,'video':video,'frames':frames,'counts':dict(counts),'strict_online_metrics':strict,'canonical_GMT_filtered_metrics':canonical,'canonical_future_length_filter':True,'strict_predictions':{'path':str(out/'RAW_PREDICTIONS.json'),'SHA256':sha(out/'RAW_PREDICTIONS.json')},'canonical_predictions':{'path':str(out/'PREDICTIONS.json'),'SHA256':sha(out/'PREDICTIONS.json')},'paired_H32_causal_audit':causal,'identity_summary':{k:v for k,v in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},'cross_camera':{k:v for k,v in identity['identity_index_query'].items() if k!='queries'},'online_candidate_lifecycle_audit':{k:v for k,v in diagnostic.items() if k not in ['wrong_ID_episodes','rows','row_statuses']},'policy_latency_ms':quantile(latencies,'policy_assignment_ms'),'extra_vs_GMT_policy_latency_ms':quantile(latencies,'extra_policy_assignment_ms'),'peak_VRAM_MiB':quantile(latencies,'peak_VRAM_MiB'),'typed_MATCH_latency_ms':quantile(stages,'total_ms'),'native_latency_ms':quantile(native,'native_association_commit_ms_excluding_paired_reference'),'elapsed_seconds':time.monotonic()-begin,'full_backbone_recomputed':False,'state_cache_cross_stage_hits':0,'within_MATCH_state_shared_across_questions':True,'lifecycle':'MEMORY/REACTIVATION frozen native fallback','actual_mutated_state_online':True,'heldout':'SEALED','Full24':False,'official_TEST':False}
    save(done,result);save(out/'PROGRESS.json',{'status':'COMPLETE','frames':frames,'strict_metrics':strict,'canonical_metrics':canonical})
    # Recovery slots are mutable work files, not historical scientific assets.
    # Causal prefix digest and actual fork results remain; deterministic actor
    # plus frozen cache/checkpoint recreates the predeclared boundary.
    for path in list(out.glob('UNFROZEN_RECOVERY_SLOT_*.pth'))+[out/'CAUSAL_PREFIX_128_0.pth']:
        if path.exists():path.unlink()
    print('NATIVE_FULL_ONLINE_COMPLETE',name,video,strict,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--case',required=True);p.add_argument('--smoke-frames',type=int,default=0);p.add_argument('--chunk-size',type=int,default=256);a=p.parse_args()
    try:run(a.video,a.case,a.smoke_frames,a.chunk_size)
    except Exception:
        save(OUT/'online_failures'/f'{a.case}_video{a.video}_{time.time_ns()}.json',{'status':'FAIL','binding':binding(),'traceback':traceback.format_exc()});raise
