"""Run official inference once, only after both actual trained stages are verified."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parents[1];repo=R/'code/GMT'
lock=(R/'manifests/inference.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for stage in [1,2]:
 record=json.loads((R/f'manifests/stage{stage}_complete.json').read_text())
 assert record['iterations']==20000 and record['scheduler_last_epoch']==20000 and record['checkpoint_reload']=='PASS'
checkpoint=Path(record['checkpoint']);h=hashlib.sha256()
with checkpoint.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==record['sha256']
for line in (R/'manifests/official_configs.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected
preflight_env=os.environ.copy();preflight_env['CUDA_VISIBLE_DEVICES']='0'
with (R/'logs/final_inference_checkpoint_preflight.log').open('xb') as log:
 subprocess.run([str(R/'tools/run_gmt.sh'),str(R/'tools/validate_inference_checkpoint.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,env=preflight_env,check=True)
preflight=json.loads((R/'manifests/final_inference_checkpoint_preflight.json').read_text())
assert preflight['status']=='PASS' and preflight['checkpoint_sha256']==record['sha256']
predictions=R/'outputs/predictions_stage2';output=R/'outputs/inference_stage2'
writer=repo/'VISIONT18000_13_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50'
for path in [predictions,output,writer,R/'manifests/inference_complete.json']:
 if path.exists() or path.is_symlink():raise RuntimeError('Refusing append/reuse inference destination: '+str(path))
predictions.mkdir();writer.symlink_to(predictions,target_is_directory=True)
args=[str(R/'tools/run_gmt.sh'),'test_net.py','--num-gpus','1','--config-file','configs/VISION_test.yaml','MODEL.WEIGHTS',str(checkpoint),'OUTPUT_DIR',str(output)]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
start=time.monotonic();state={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':str(checkpoint),'checkpoint_sha256':h.hexdigest(),'predictions':str(predictions),'official_writer_symlink':str(writer),'args':args,'status':'running'}
with (R/'logs/inference_stage2.log').open('xb') as log:
 process=subprocess.Popen(args,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env);state['pid']=process.pid
 (R/'manifests/inference_state.json').write_text(json.dumps(state,indent=2));rc=process.wait()
state.update(exit_code=rc,duration_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='exited')
(R/'manifests/inference_state.json').write_text(json.dumps(state,indent=2))
if rc:raise RuntimeError('Official inference failed; preserve partial predictions, do not append on retry')
names=(repo/'TrackEval/data/gt/mot_challenge/seqmaps/vision-train.txt').read_text().splitlines()[1:]
expected={str(Path(n.rsplit('_',1)[0])/(n.rsplit('_',1)[1]+'.txt')) for n in names}
actual={str(p.relative_to(predictions)) for p in predictions.rglob('*.txt')}
assert expected==actual,(expected-actual,actual-expected)
state.update(status='complete_pending_prediction_validation',views=len(actual),scenes=len({n.rsplit('_',1)[0] for n in names}))
(R/'manifests/inference_complete.json').write_text(json.dumps(state,indent=2));print('TRAINED_STAGE2_INFERENCE_COMPLETE',flush=True)
