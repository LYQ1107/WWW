"""Fixed execution-profile factorial on the original12teacher groups, no new model."""
from jev_phase21_common import *
import gc,statistics,contextlib,collections
import torch
from PIL import Image
EXTERNAL=TEACHER/'external/Visual-Jev'
sys.path.insert(0,str(EXTERNAL/'code'))
from jev_phase17_token_verified_vdm import TokenVerifiedVDM


def probabilities(o,questions):
 return [o['lm_option_logits'][i,:len(q['candidates'])].softmax(-1).cpu().tolist() for i,q in enumerate(questions)]


def main():
 protect();started=time.monotonic();assert not(OUT/'RESULT.json').exists()
 pre=read(REPORTS/'PREREGISTRATION.json');assert sha(pre['population']['path'])==pre['population']['SHA256']
 pairs=read(pre['population']['path']);chosen={}
 for pair in pairs:chosen.setdefault(pair['inference']['group'],pair['inference'])
 assert len(chosen)==12
 torch.set_num_threads(1);torch.manual_seed(20261009)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 models=read(TEACHER/'teacher_models/MANIFEST.json');tests=[];latencies=[];verified=[]
 for model in models:
  for f in model['files']:
   p=Path(f['path'])
   if p.name.endswith('.safetensors') and p.parent.name not in ['seed1','seed2']:
    save(OUT/'PROGRESS.json',dict(status='VERIFYING_FROZEN_WEIGHTS',path=str(p),seconds=time.monotonic()-started))
    assert sha(p)==f['SHA256'];verified.append(f)
 def guard():
  protect();assert time.monotonic()-started<1800,'preregistered wall limit reached'
 for profile in pre['profiles']:
  guard();gc.collect();torch.cuda.empty_cache()
  free,_=torch.cuda.mem_get_info();assert free/2**20>=26000,'insufficient free VRAM for fixed FP32 admission budget'
  torch.backends.cuda.enable_flash_sdp(profile['backend']=='default_sdpa')
  torch.backends.cuda.enable_mem_efficient_sdp(profile['backend']=='default_sdpa')
  torch.backends.cuda.enable_math_sdp(True)
  dtype=getattr(torch,profile['dtype'])
  save(OUT/'PROGRESS.json',dict(status='LOADING_MODEL',profile=profile,seconds=time.monotonic()-started))
  model=TokenVerifiedVDM(models[0]['folder'],dtype=dtype,with_heads=False)
  model.processor.image_processor.max_pixels=65536;model.processor.image_processor.min_pixels=3136
  assert model.heads is None
  for variant in pre['variants']:
   if variant=='published_answer_LoRA':
    from peft import PeftModel
    model.backbone=PeftModel.from_pretrained(model.backbone,models[1]['folder'],is_trainable=False).eval()
   for param in model.parameters():param.requires_grad_(False)
   for wire in chosen.values():
    guard();assert sha(wire['image']['path'])==wire['image']['SHA256']
    image=Image.open(wire['image']['path']).convert('RGB');q=wire['questions']
    g=model.prepare_group(image,q,wire['shared_context']);assert g.token_boundaries_exact
    pos=all(torch.equal(g.position_ids[0][:,:g.prefix_len],p[:,:g.prefix_len]) for p in g.position_ids)
    assert pos
    base=model.run_independent(g);bp=probabilities(base,q);paths=[]
    for name,method in [('vision_cache',model.run_vision_cache),('independent_batch',model.run_independent_batch),('prefix_share',model.run_prefix_share),('prefix_share_batch',model.run_prefix_share_batch)]:
     o=method(g);op=probabilities(o,q)
     mz=max(float((base['lm_option_logits'][i,:len(z['candidates'])]-o['lm_option_logits'][i,:len(z['candidates'])]).abs().max()) for i,z in enumerate(q))
     mp=max(abs(a-b) for p,z in zip(bp,op) for a,b in zip(p,z))
     same=all(max(range(len(a)),key=a.__getitem__)==max(range(len(b)),key=b.__getitem__) for a,b in zip(bp,op))
     paths.append(dict(path=name,max_abs_valid_logit_delta=mz,max_abs_probability_delta=mp,decisions_exact=same,PASS=mz<=.05 and mp<=.02 and same))
    reverse=dict(q[0],candidates=list(reversed(q[0]['candidates'])))
    rg=model.prepare_group(image,[reverse],wire['shared_context']);rp=list(reversed(probabilities(model.run_independent(rg),[reverse])[0]))
    order_same=max(range(len(rp)),key=rp.__getitem__)==max(range(len(bp[0])),key=bp[0].__getitem__)
    cache=model._prefill_prefix(g);fork=model._fork(cache)
    original=cache.layers[0].keys.clone();ptrs=all(c.keys.data_ptr()!=f.keys.data_ptr() and c.values.data_ptr()!=f.values.data_ptr() for c,f in zip(cache.layers,fork.layers))
    fork.layers[0].keys.zero_();noalias=ptrs and torch.equal(original,cache.layers[0].keys)
    assert noalias
    tests.append(dict(profile=profile['name'],variant=variant,group=wire['group'],event_key=wire['GT_free_event_key'],image=wire['image'],
     paths=paths,option_reverse_decision_exact=order_same,option_reverse_probability_delta=max(abs(a-b) for a,b in zip(bp[0],rp)),
     token_boundary_exact=True,prefix_positions_identical=pos,KV_fork_no_alias=noalias,
     parameter_dtypes=sorted(set(str(p.dtype) for p in model.parameters())),full_forward_probabilities=bp))
    del cache,fork,original,base,o,g,rg
    save(OUT/'PARTIAL_TESTS.json',tests)
    save(OUT/'PROGRESS.json',dict(status='RUNNING',profile=profile['name'],variant=variant,completed_tests=len(tests),total_tests=72,group=wire['group'],seconds=time.monotonic()-started,pid=os.getpid()))
    print('VISUAL_PARITY_GROUP',profile['name'],variant,wire['group'],flush=True)
   if profile['name'] in pre['latency']['profiles']:
    wire=next(iter(chosen.values()));image=Image.open(wire['image']['path']).convert('RGB')
    for n in pre['latency']['N']:
     guard();q=[wire['questions'][i%len(wire['questions'])] for i in range(n)]
     t=time.perf_counter();g=model.prepare_group(image,q,wire['shared_context']);prepare=(time.perf_counter()-t)*1000
     for name,method in [('independent_serial',model.run_independent),('independent_batch',model.run_independent_batch),('shared_prefix_batch',model.run_prefix_share_batch)]:
      guard();method(g);dur=[];torch.cuda.reset_peak_memory_stats()
      for _ in range(2):
       torch.cuda.synchronize();t=time.perf_counter();method(g);torch.cuda.synchronize();dur.append((time.perf_counter()-t)*1000)
      latencies.append(dict(profile=profile['name'],variant=variant,N=n,path=name,total_batch_ms=dur,p50_total_ms=statistics.median(dur),max_of2_total_ms=max(dur),
       amortized_ms_per_question=[x/n for x in dur],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,prepare_ms=prepare,p95_estimate=None))
     torch.cuda.synchronize();t=time.perf_counter();cache=model._prefill_prefix(g);torch.cuda.synchronize();prefill=(time.perf_counter()-t)*1000
     t=time.perf_counter();fork=model._fork(cache,n);torch.cuda.synchronize();fork_ms=(time.perf_counter()-t)*1000
     latencies.append(dict(profile=profile['name'],variant=variant,N=n,path='prefill_and_fork_components',prefill_ms=prefill,KV_fork_ms=fork_ms,full_execution_ms=None))
     del cache,fork,g
     save(OUT/'PARTIAL_LATENCIES.json',latencies)
     save(OUT/'PROGRESS.json',dict(status='LATENCY_STRESS',profile=profile['name'],variant=variant,N=n,completed_tests=len(tests),seconds=time.monotonic()-started,pid=os.getpid()))
     print('VISUAL_LATENCY_DONE',profile['name'],variant,n,flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 assert len(tests)==72
 summaries={}
 for p in pre['profiles']:
  ts=[t for t in tests if t['profile']==p['name']]
  summaries[p['name']]=dict(group_variants=len(ts),all_paths_pass=all(a['PASS'] for t in ts for a in t['paths']),
   path_passes={k:sum(a['PASS'] for t in ts for a in t['paths'] if a['path']==k) for k in pre['paths']},
   max_logit_delta=max(a['max_abs_valid_logit_delta'] for t in ts for a in t['paths']),
   max_probability_delta=max(a['max_abs_probability_delta'] for t in ts for a in t['paths']),
   order_exact=sum(t['option_reverse_decision_exact'] for t in ts),order_total=len(ts))
 result=dict(status='COMPLETE_FIXED_EXECUTION_AUDIT',binding=binding(),summary=summaries,tests=tests,latencies=latencies,
  verified_existing_weights=verified,models=ref(TEACHER/'teacher_models/MANIFEST.json'),
  external_source=ref(EXTERNAL/'code/vdm/models/vdm_model.py'),token_adapter=ref(ROOT/'reproduction_tools/jev_phase17_token_verified_vdm.py'),
  environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),python=sys.version),
  seconds=time.monotonic()-started,teacher_identity_gain='NOT_REEVALUATED',new_training=False,new_native_actions=0,
  limitations=['Option presentation changes prompt semantics; it is separate from numerical cache parity.',
   'Same original24groupvariants, not independent new data.', 'RepeatedquestionN32stress does not score32dynamicidentitycandidates.',
   'Two timing samples cannot estimate p95; decodedmontage inference excludes fileIO/serving/nativeMOT.',
   'No new model weights, no fair trained sameimage controls or identity qualification.'])
 save(OUT/'RESULT.json',result);save(REPORTS/'VISUAL_EXECUTION_PROFILE_PARITY.json',result)
 save(OUT/'PROGRESS.json',dict(status='COMPLETE',result=ref(OUT/'RESULT.json')))
 protect();print(json.dumps(summaries,indent=2))


if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(OUT/'FAILURE.json',dict(status='EXECUTION_FAILURE',type=type(exc).__name__,message=str(exc),binding=binding(),partial_tests=ref(OUT/'PARTIAL_TESTS.json') if (OUT/'PARTIAL_TESTS.json').exists() else None))
  raise
