"""Distinct adapted GTA score diagnostic on immutable existing native prefixes."""
import os,sys,json,hashlib,time,subprocess,shutil,gzip,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/liuyeqiang/WWW_jev_phase21')
for p in [SOURCE,SOURCE/'reproduction_tools',SOURCE/'third_party/CenterNet2']:sys.path.insert(0,str(p))
OUT=Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/20261011_v1');OUT.mkdir(parents=True,exist_ok=True)
REPORTS=ROOT/'reports/JEV_MASTER'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p))
def save(p,x):
 p=Path(p);assert p.is_relative_to(OUT) or p.is_relative_to(REPORTS)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def binding():return dict(actor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),actor_source=ref(Path(__file__)),protocol=ref(REPORTS/'COMMON_CACHE_GTA_PROTOCOL.json'),source_root=str(SOURCE),source_root_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip())

def geometry(inst,view):
 """Official RPCE geometric formula, explicitly without original ROI jitter."""
 import torch
 b=inst.pred_boxes.tensor;n=len(inst);k=4
 scale=b.new_tensor([inst.image_size[1],inst.image_size[0]])
 centres=(b[:,:2]+b[:,2:])/2;diff=centres.unsqueeze(0)-centres.unsqueeze(1)
 dist=torch.norm(diff,dim=-1);dist.fill_diagonal_(float('inf'))
 _,indices=torch.topk(dist,min(k,max(0,n-1)),largest=False,dim=1)
 out=b.new_zeros((n,11));out[:,:2]=centres/scale
 radii=5*torch.sqrt((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))
 for i in range(n):
  count=min(k,int((dist[i]<=float(radii[i])).sum()))
  if count:out[i,2:2+2*count]=(diff[i,indices[i,:count]]/scale).flatten()
 out[:,-1]=view/2
 return out

def main():
 import torch,numpy as np,random
 from jev_phase14_artifacts import load_dense
 from detectron2.config import get_cfg
 from detectron2.modeling import build_model
 from centernet.config import add_centernet_config
 from gtr.config import add_gtr_config
 from gtr.modeling.roi_heads.transformer import TrajectoryRandom
 start=time.monotonic();assert not (OUT/'RESULT.json').exists()
 pre=read(REPORTS/'COMMON_CACHE_GTA_PROTOCOL.json');assert sha(pre['population']['path'])==pre['population']['SHA256']
 assert binding()['source_root_commit']==pre['source_root_commit']
 torch.set_num_threads(1);torch.manual_seed(20261009)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 def guard():
  assert time.monotonic()-start<pre['budgets']['wall_seconds']
  assert shutil.disk_usage(OUT).free>=30*2**30
  assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=32*2**20
 guard();assert torch.cuda.mem_get_info()[0]/2**20>=8000
 save(OUT/'PROGRESS.json',dict(status='VERIFYING_EXISTING_ASSOCIATION_WEIGHTS',pid=os.getpid()))
 assert sha(pre['scorer_checkpoint']['path'])==pre['scorer_checkpoint']['SHA256']
 cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(SOURCE/pre['model_config']));cfg.defrost();cfg.MODEL.JEV.ENABLED=False;cfg.freeze()
 model=build_model(cfg).cuda().eval();checkpoint=torch.load(pre['scorer_checkpoint']['path'],map_location='cpu')
 model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
 for p in model.parameters():p.requires_grad_(False)
 assert model.roi_heads.concat_emb and model.roi_heads.cat_dim==128 and model.roi_heads.feature_dim==1024
 cases=[];source_functions=[SOURCE/'gtr/modeling/meta_arch/gtr_rcnn.py',SOURCE/'gtr/modeling/roi_heads/gtr_roi_heads.py',SOURCE/'gtr/modeling/roi_heads/transformer.py',SOURCE/'reproduction_tools/cache_jev_phase13_stage1.py']
 events=read(pre['population']['path']);assert len(events)==12
 for index,event in enumerate(events):
  guard();assert sha(event['prefix']['path'])==event['prefix']['SHA256'];assert sha(event['source_state']['path'])==event['source_state']['SHA256']
  prefix=load_dense(event['prefix']['path'],map_location='cpu');raw=load_dense(event['source_state']['path'],map_location='cpu')['raw_batch']
  video,frame,view,row=event['GT_free_event_key'];assert prefix['key']==[video,frame,view] and not prefix['first']
  begin=max(0,frame+1-model.test_len)*2;end=frame*2+view
  assert len(prefix['instances'])==end+1
  originals=prefix['instances'][begin:end+1];batch=[]
  for offset,inst in enumerate(originals):
   key=divmod(begin+offset,2);assert key<=tuple([frame,view])
   if offset<len(originals)-1:assert key<(frame,view) and inst.has('track_ids')
   assert inst.reid_features.shape==(len(inst),1024)
   item=inst.to('cuda');item.reid_features=inst.reid_features.to('cuda').clone()
   with torch.no_grad():item.reid_features=torch.cat([item.reid_features,model.roi_heads.s_t_head(geometry(item,key[1]))],dim=1)
   batch.append(item)
  assert torch.equal(batch[-1].reid_features[:,:1024].cpu(),raw['detection_visual'][0])
  actual=read(Path(event['source_state']['path']).parent/'RESULT.json')
  with gzip.open(actual['queries']['path'],'rt') as f:q=json.loads(f.readline())
  assert q['task']==0 and q['context']['frame']==frame and q['context']['view']==view
  refs=q['refs'];past_ids=torch.cat([x.track_ids for x in batch[:-1]])
  unique=torch.unique(past_ids);assert unique.tolist()==refs
  legal=raw['legal'][0];assert legal.shape==(len(batch[-1]),len(refs))
  outputs=[];durations=[];rngs=[]
  for repeat in range(2):
   model._jev_trajectory_rng=TrajectoryRandom();model._jev_trajectory_rng.setstate(prefix['trajectory_rng_state'])
   model._jev_trajectory_rng._trajectory_draws=copy.deepcopy(prefix['trajectory_rng_draws'])
   random.setstate(prefix['python_rng']);np.random.set_state(prefix['numpy_rng']);torch.set_rng_state(prefix['torch_rng']);torch.cuda.set_rng_state(prefix['cuda_rng'])
   torch.cuda.synchronize();t=time.perf_counter()
   with torch.no_grad():
    assoc,_,counts,npast,_=model.get_asso(batch,len(batch)-1)
    assert npast==len(past_ids)
    activated=model.roi_heads._activate_asso(assoc[-1].split(counts[:-1],dim=1))
    scores=torch.cat(activated,dim=1)@((unique[None,:]==past_ids[:,None]).float())
   torch.cuda.synchronize();durations.append((time.perf_counter()-t)*1000)
   assert torch.isfinite(scores).all();outputs.append(scores.cpu());rngs.append(copy.deepcopy(model._jev_trajectory_rng._trajectory_draws))
  delta=float((outputs[0]-outputs[1]).abs().max());assert delta==0 and rngs[0]==rngs[1]
  scores=outputs[0][row];legal_refs=[ref_id for j,ref_id in enumerate(refs) if legal[row,j]]
  assert legal_refs==event['legal_refs'] and legal_refs
  order=sorted(legal_refs,key=lambda ref_id:(-float(scores[refs.index(ref_id)]),ref_id))
  cosine=sorted(legal_refs,key=lambda ref_id:(-float(raw['pair_evidence'][0,row,refs.index(ref_id),:4].max()),ref_id))
  # Labels read only after every score, ranking and prediction has been fixed.
  owners={c['identity']:c['qualified_owner_GT_OFFLINE_ONLY'] for c in event['candidates']}
  gt=event['GT_OFFLINE_ONLY'];picks={'common_cache_GTA':order[0],'four_slot_cosine':cosine[0],'frozen_XV_JEV':event['selected_ref']}
  evaluated={k:dict(ref=v,owner_GT_OFFLINE=owners.get(v),outcome='UNKNOWN' if owners.get(v) is None else 'CORRECT' if owners[v]==gt else 'WRONG') for k,v in picks.items()}
  cases.append(dict(group=index,event_key=event['GT_free_event_key'],prefix=event['prefix'],source_state=event['source_state'],
   refs=refs,legal_refs=legal_refs,target_row=row,raw_GTA_scores=scores.tolist(),ranked_legal_refs=order,old_correct_owner_refs_OFFLINE=event['certified_correct_available'],
   rank_of_correct_owner=min([order.index(v)+1 for v in event['certified_correct_available'] if v in order],default=None),
   predictions=evaluated,prefix_VFCE_unchanged=True,current_VFCE_matches_original_query=True,repeat_max_score_delta=delta,
   trajectory_draws=rngs[0],timing_ms=durations,window_start_payload=begin,window_payload_count=len(batch),
   GTA_input_dim=1152,available_cache_dim=1024,geometry_no_jitter=True,actual_source_camera_feature=True))
  save(OUT/'PARTIAL_CASES.json',cases);save(OUT/'PROGRESS.json',dict(status='RUNNING',group=index,completed=len(cases),total=12,pid=os.getpid(),seconds=time.monotonic()-start))
  print('COMMON_CACHE_GTA_GROUP',index,event['GT_free_event_key'],evaluated,flush=True)
  del batch,prefix,raw,outputs
 summary={name:{outcome:sum(c['predictions'][name]['outcome']==outcome for c in cases) for outcome in ['CORRECT','WRONG','UNKNOWN']} for name in ['common_cache_GTA','four_slot_cosine','frozen_XV_JEV']}
 result=dict(status='COMPLETE_ADAPTED_COMMON_CACHE_SCORE_DIAGNOSTIC',binding=binding(),cases=cases,summary=summary,
  source_files=[ref(p) for p in source_functions],checkpoint=pre['scorer_checkpoint'],model_config=ref(SOURCE/pre['model_config']),seconds=time.monotonic()-start,
  peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,environment=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),
  original_GMT_D0='NOT_RECONSTRUCTED',new_training=False,new_native_branches=0,causal_corrections=None,independent_value='NOT_ESTABLISHED',
  limitations=['Explicit unjittered common-cache RPCE/GTA adapter, not original GMT frontend/policy replay.',
   'Point-wise top rank, not joint assignment, actual commit or long-horizon intervention.',
   'FrozenXVII pure-history owner diagnostic, not newer immutable-owner safe training certification.',
   'Same12previouslyused teacher groups, old outcomes known; no independent generalization/significance.',
   'Recent numerical history and three-panel visual teacher evidence differ; no architecture superiority claim.'])
 guard();save(OUT/'RESULT.json',result);save(REPORTS/'COMMON_CACHE_GTA_RESULTS.json',result);save(OUT/'PROGRESS.json',dict(status='COMPLETE',result=ref(OUT/'RESULT.json')))
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 try:main()
 except Exception as e:
  save(OUT/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='ENGINEERING_FAILURE',type=type(e).__name__,message=str(e),binding=binding()));raise
