"""No-new-head teacher: actual crops, all legal candidates, measured execution parity."""
from jev_phase17_common import *
import argparse,copy,gzip,statistics
import torch
from PIL import Image
external=OUT/'external/Visual-Jev';sys.path.insert(0,str(external/'code'))
from jev_phase17_token_verified_vdm import TokenVerifiedVDM as VDM

def probabilities(output,questions):
 return [output['lm_option_logits'][i,:len(q['candidates'])].softmax(-1).float().cpu().tolist() for i,q in enumerate(questions)]
def execute():
 protect();assert read(REPORTS/'VISUAL_TEACHER_PROTOCOL.json')['no_new_head'];torch.set_num_threads(1);torch.manual_seed(20261009)
 models=read(OUT/'teacher_models/MANIFEST.json');base=models[0];data=read(OUT/'teacher_data/PAIRS.json');folder=OUT/'teacher_inference_v2';folder.mkdir(parents=True,exist_ok=True);allresults=[];parity=[];latency=[]
 model=VDM(base['folder'],dtype=torch.float16,with_heads=False);model.processor.image_processor.max_pixels=65536;model.processor.image_processor.min_pixels=3136
 assert model.heads is None
 for variant in ['pretrained','published_answer_LoRA']:
  if variant=='published_answer_LoRA':
   from peft import PeftModel
   model.backbone=PeftModel.from_pretrained(model.backbone,models[1]['folder'],is_trainable=False).eval()
  stream=gzip.open(folder/(variant+'.jsonl.gz'),'wt');records=[];seen_groups=set();boundary_checks=[];started=time.monotonic()
  for number,pair in enumerate(data):
   wire=pair['inference'];assert sha(wire['image']['path'])==wire['image']['SHA256'];image=Image.open(wire['image']['path']).convert('RGB');questions=wire['questions'];g=model.prepare_group(image,questions,wire['shared_context'])
   with torch.no_grad():
    output=model.run_independent(g)
   boundary_checks.append(dict(pair=number,exact=g.token_boundaries_exact,removed_public_prefix_tokens=g.public_prefix_tokens_removed));probs=probabilities(output,questions);item=dict(group=wire['group'],event_key=wire['GT_free_event_key'],candidate_identity=wire['candidate_identity'],questions=[q['qid'] for q in questions],logits=output['lm_option_logits'].float().cpu().tolist(),probabilities=probs,evaluation=pair['evaluation'],source_image=wire['image'])
   if wire['group'] not in seen_groups:
    seen_groups.add(wire['group']);ind=model.run_independent(g);serial=model.run_prefix_share(g);batch=model.run_prefix_share_batch(g);pbase=probabilities(ind,questions)
    pd=[]
    for name,other in [('independent_batch',model.run_independent_batch(g)),('prefix_share',serial),('prefix_share_batch',batch)]:
     pother=probabilities(other,questions);maxp=max(abs(a-b) for p,q in zip(pbase,pother) for a,b in zip(p,q));maxz=max(float((ind['lm_option_logits'][i,:len(q['candidates'])]-other['lm_option_logits'][i,:len(q['candidates'])]).abs().max()) for i,q in enumerate(questions));same=all(max(range(len(a)),key=a.__getitem__)==max(range(len(b)),key=b.__getitem__) for a,b in zip(pbase,pother));pd.append(dict(path=name,max_abs_valid_logit_delta=maxz,max_abs_probability_delta=maxp,decisions_exact=same,PASS=maxz<=.05 and maxp<=.02 and same))
    reverse=dict(questions[0],candidates=list(reversed(questions[0]['candidates'])));rg=model.prepare_group(image,[reverse],wire['shared_context']);ro=model.run_independent(rg);rp=list(reversed(probabilities(ro,[reverse])[0]));order_delta=max(abs(a-b) for a,b in zip(pbase[0],rp));order_same=max(range(len(rp)),key=rp.__getitem__)==max(range(len(pbase[0])),key=pbase[0].__getitem__)
    prefix_equal=all(torch.equal(g.position_ids[0][:,:g.prefix_len],position[:,:g.prefix_len]) for position in g.position_ids)
    cache=model._prefill_prefix(g);original=[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers];fork=model._fork(cache);first_layer=fork.layers[0];first_layer.keys.zero_();noalias=all(torch.equal(layer.keys,kv[0]) and torch.equal(layer.values,kv[1]) for layer,kv in zip(cache.layers,original))
    parity.append(dict(variant=variant,group=wire['group'],source_image=wire['image'],paths=pd,option_reverse_probability_delta=order_delta,option_reverse_decision_exact=order_same,prefix_positions_identical=prefix_equal,KV_fork_no_alias=noalias,cross_frame_cache_reuse=False,public_context_has_other_answers=False));del cache,fork,original
   records.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n')
   if number%16==0:
    stream.flush();save(folder/'PROGRESS.json',dict(status='RUNNING',variant=variant,completed_pairs=number+1,total_pairs=len(data),completed_groups=len(seen_groups),seconds=time.monotonic()-started,token_boundary_checks=boundary_checks));print('TEACHER_PAIR_PROGRESS',variant,number+1,len(data),flush=True)
  stream.close();allresults.append(dict(variant=variant,pairs=len(records),records=ref(folder/(variant+'.jsonl.gz')),seconds=time.monotonic()-started,token_boundary_checks=boundary_checks))
  wire=data[0]['inference'];image=Image.open(wire['image']['path']).convert('RGB')
  for n in [1,4,8,16,32]:
   questions=[wire['questions'][i%len(wire['questions'])] for i in range(n)];t0=time.perf_counter();g=model.prepare_group(image,questions,wire['shared_context']);prepare_ms=(time.perf_counter()-t0)*1000
   for name,method in [('independent_serial',model.run_independent),('independent_batch',model.run_independent_batch),('shared_prefix_batch',model.run_prefix_share_batch)]:
    method(g);dur=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
     torch.cuda.synchronize();t0=time.perf_counter();method(g);torch.cuda.synchronize();dur.append((time.perf_counter()-t0)*1000)
    latency.append(dict(variant=variant,N=n,path=name,total_batch_ms=dur,amortized_ms_per_question=[v/n for v in dur],p50_total_batch_ms=statistics.median(dur),p95_total_batch_ms=sorted(dur)[-1],p50_amortized_ms=statistics.median(dur)/n,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,prepare_ms=prepare_ms,scope='decoded real crop montage, processor preparation separately measured, full vision/prefix/suffix/readout execution; excludes disk decoding and serving queue; diagnostic not online GMT FPS'))
  # K variations include identical-evidence diagnostic irrelevant alternatives; never tracking candidate truncation.
  counts=[]
  for k in [2,3,4,8,16]:
   q=dict(wire['questions'][0],candidates=['same person','different person']+['insufficient or mixed evidence']*(k-2));g=model.prepare_group(image,[q],wire['shared_context']);o=model.run_independent(g);counts.append(dict(K=k,probabilities=probabilities(o,[q])[0],scope='readout-count stress with duplicate uncertainty choices; not identity candidate accuracy',duplicate_choices=True))
  save(folder/(variant+'_COUNT_STRESS.json'),counts)
 result=dict(status='COMPLETE',binding=binding(),variants=allresults,models=ref(OUT/'teacher_models/MANIFEST.json'),external_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=external,text=True).strip(),new_head=False,new_identity_training=False,scoring_path='independent_serial_full_forward',actual_actor_commit=os.environ['PHASE17_ACTOR_COMMIT'],actual_token_boundaries_verified=True,failed_original_attempt=ref(OUT/'teacher_inference/RESULT.json'),groups=len({p['inference']['group'] for p in data}),pairs=len(data),environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),dtype='float16',python=sys.version),parity=parity,latency=latency,all_legal_candidates_scored=True,global_unvalidated_softmax_ranking=False)
 save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status='COMPLETE',result=ref(folder/'RESULT.json')));print('TEACHER_COMPLETE',flush=True)
if __name__=='__main__':execute()
