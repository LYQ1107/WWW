"""Authorized gated Stage1/Stage2 training; stop on any failed gate, never tune."""
from pathlib import Path
import os,sys,json,time,datetime,subprocess,fcntl,hashlib,math,shlex,statistics
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');statefile=r/'manifests/training_pipeline_state.json'
WORLD_SIZE=int(os.environ.get('GMT_WORLD_SIZE','4')); BATCH_SIZE=int(os.environ.get('GMT_BATCH_SIZE','4'))
assert WORLD_SIZE==4 and BATCH_SIZE==4 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3'
lock=(r/'manifests/training_pipeline.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'current':'starting','jobs':[]}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(current,**extra):
 state.update(current=current,updated_utc=now(),**extra);statefile.write_text(json.dumps(state,indent=2));print(current,extra,flush=True)
 p=r/'STATUS.md';s=p.read_text();a=s.index('Current milestone:');b=s.index('Completed:');s=s[:a]+'Current milestone:\n'+current+'\n\n'+s[b:]
 a=s.index('Running:');b=s.index('Blocked:');s=s[:a]+'Running:\nTraining controller PID '+str(os.getpid())+'; state: '+current+'. See manifests/training_pipeline_state.json for current child PID and commands.\n\n'+s[b:]
 s=s[:s.index('Updated UTC:')]+'Updated UTC:\n'+now()+'\n\nUser requests autonomous continuation without further questions. Approved: four GPUs0–3, batch4, repository20k+20k/AdamW/LR settings; preserve official model/evaluation algorithms.\n';p.write_text(s)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat';return p.exists() and p.read_text().split()[2]!='Z'
def run(label,args,log,env=None):
 with (r/'reports/commands_used.sh').open('a') as f:f.write('\n# '+label+'\n'+shlex.join(args)+'\n')
 with (r/'logs'/log).open('ab') as f:
  started=time.monotonic();proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
  job={'label':label,'pid':proc.pid,'started_utc':now(),'args':args,'log':str(r/'logs'/log)};state['jobs'].append(job);save(label,child_pid=proc.pid)
  with (r/'reports/running_jobs.md').open('a') as report:report.write('\n'+json.dumps(job)+'\n')
  rc=proc.wait();job.update(exit_code=rc,duration_seconds=time.monotonic()-started,finished_utc=now());save(label+'_exited',child_pid=None)
 if rc!=0:raise RuntimeError(label+' exited '+str(rc)+', inspect '+log)
 return job

def gate_stage(stage,job):
 import torch
 out=r/f'outputs/stage{stage}';cp=out/'model_20000.pth'
 if not cp.is_file():raise RuntimeError('Expected 20k checkpoint missing')
 rows=[json.loads(line) for line in (out/'metrics.json').read_text().splitlines() if line.strip()]
 if not rows or int(rows[-1]['iteration'])!=20000:raise RuntimeError('Full 20000 iterations not proven')
 loss=[float(row['total_loss']) for row in rows if 'total_loss' in row]
 if not loss or not all(math.isfinite(x) for x in loss):raise RuntimeError('Nonfinite/missing total loss')
 for row in rows:
  for k,v in row.items():
   if 'loss' in k and isinstance(v,(int,float)) and not math.isfinite(v):raise RuntimeError('Nonfinite component loss')
 ckpt=torch.load(cp,map_location='cpu')
 if not all(k in ckpt for k in ['model','optimizer','scheduler']):raise RuntimeError('Incomplete checkpoint')
 if not all(torch.isfinite(v).all() for v in ckpt['model'].values() if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())):raise RuntimeError('Nonfinite model parameters')
 epoch=ckpt['scheduler'].get('last_epoch')
 if epoch!=20000:raise RuntimeError('Scheduler completion mismatch: '+str(epoch))
 first=statistics.median(loss[:min(20,len(loss))]);last=statistics.median(loss[-min(20,len(loss)):])
 if last>10*max(first,1e-8):raise RuntimeError('Large sustained loss growth at completed stage; inspect before proceeding')
 h=hashlib.sha256()
 with cp.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 record={'stage':stage,'iterations':20000,'scheduler_last_epoch':epoch,'checkpoint':str(cp),'sha256':h.hexdigest(),'bytes':cp.stat().st_size,'checkpoint_reload':'PASS','loss_finiteness':'PASS','logged_loss_samples':len(loss),'first20_logged_loss_median':first,'last20_logged_loss_median':last,'job':job}
 (r/f'manifests/stage{stage}_complete.json').write_text(json.dumps(record,indent=2))
 link=r/f'checkpoints/stage{stage}/model_20000.pth'
 if link.exists():raise RuntimeError('Refusing to replace existing final checkpoint link')
 link.symlink_to(cp)
 with (r/'manifests/checkpoints.tsv').open('a') as f:f.write(f'stage{stage}_final\t{cp}\t{cp.stat().st_size}\t{h.hexdigest()}\t\n')
 filename='05_stage1.md' if stage==1 else '08_stage2.md'
 (r/'reports'/filename).write_text('# Stage '+str(stage)+' completed\n\n```json\n'+json.dumps(record,indent=2)+'\n```\n\nFull official20k configuration with user-approved batch4/four GPUs0–3. All logged loss values finite; loss medians shown for review. No test GT tuning. Final checkpoint deserialized and all model tensors checked finite; subsequent Stage2 smoke checks actual Stage1 weight compatibility. Official trainer does not store iteration metadata in its normal checkpoints; completion is verified via metrics and scheduler state.\n')
 del ckpt
 save(f'M{12+stage}: Stage{stage} 20000 iterations verified')

try:
 save('Waiting for verified dataset and official preprocessing')
 while True:
  p=r/'manifests/data_preparation_state.json'
  d=json.loads(p.read_text()) if p.exists() else {}
  if d.get('status')=='complete_pending_loader_smoke':break
  meta=json.loads((r/'manifests/data_preparation_job.json').read_text())
  if d.get('status')=='failed' or not alive(meta['pid']):raise RuntimeError('Data preparation failed or terminated before success')
  time.sleep(15)
 # Existing helper/runtime proofs are prerequisites, not replacements for real training smoke.
 if 'ENVIRONMENT_SMOKE_PASS' not in (r/'logs/environment_smoke.log').read_text():raise RuntimeError('Environment gate missing')
 comm_gate=json.loads((r/'manifests/distributed_large_gloo_four_gpu.json').read_text())
 if comm_gate.get('status')!='PASS' or comm_gate.get('backend')!='gloo' or comm_gate.get('world_size')!=WORLD_SIZE:raise RuntimeError('Final shared-GPU communication gate missing')
 # Verify official configurations were not altered since audit.
 for line in (r/'manifests/official_configs.sha256').read_text().splitlines():
  expected,path=line.split('  ',1)
  if hashlib.sha256((r/'code/GMT'/path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Official config changed after audit')
 for stage in [1,2]:
  out=r/f'outputs/stage{stage}'
  if out.exists() and any(out.iterdir()):raise RuntimeError('Existing formal output; inspect before starting duplicate training')
  with (r/'logs'/f'gpu_before_stage{stage}.log').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,stderr=subprocess.STDOUT,check=True)
  env=os.environ.copy();env['GMT_SMOKE_STAGE']=str(stage)
  run(f'Stage{stage}_20_iteration_smoke',[str(r/'tools/run_gmt.sh'),str(r/'tools/run_training_smoke.py')],f'smoke_stage{stage}.log',env)
  variant=env.get('GMT_SMOKE_VARIANT','')
  suffix='_'+variant if variant else ''
  proof=json.loads((r/f'manifests/stage{stage}_smoke{suffix}_pass.json').read_text())
  if proof.get('world_size')!=WORLD_SIZE or proof.get('batch_size')!=BATCH_SIZE or proof.get('stage')!=stage or proof.get('iterations')!=20 or proof.get('checkpoint_reload')!='PASS' or proof.get('model_finiteness')!='PASS':raise RuntimeError('Current runtime smoke proof invalid')
  if not (Path(proof['output'])/'smoke_final.pth').is_file():raise RuntimeError('Current smoke checkpoint missing')
  weight=r/('checkpoints/backbone/CH_FPN_1x_key_adapted.pth' if stage==1 else 'outputs/stage1/model_20000.pth')
  args=[str(r/'tools/run_gmt.sh'),str(r/'tools/run_observed_training.py'),'--num-gpus',str(WORLD_SIZE),'--config-file',f'configs/VISION_stage{stage}.yaml','SOLVER.IMS_PER_BATCH',str(BATCH_SIZE),'MODEL.WEIGHTS',str(weight),'OUTPUT_DIR',str(out)]
  job=run(f'Stage{stage}_formal_20000_iterations',args,f'train_stage{stage}.log')
  gate_stage(stage,job)
 save('Training stages complete; inference/evaluation preparation required')
except Exception as e:
 save('FAILED: do not restart automatically',error=str(e));raise
