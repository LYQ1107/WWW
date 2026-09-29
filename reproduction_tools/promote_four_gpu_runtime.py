"""Wait for the exact four-GPU diagnostic; launch fresh smoke/full training only on proof."""
from pathlib import Path
import datetime, fcntl, hashlib, json, math, os, shlex, shutil, subprocess, time
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'manifests/four_gpu_promotion_plan.json'
STATE = ROOT/'manifests/four_gpu_promotion_state.json'
lock = (ROOT/'manifests/four_gpu_promotion.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
plan = json.loads(PLAN.read_text())
state = {'pid': os.getpid(), 'diagnostic_pid': plan['diagnostic_pid']}
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(status, **extra):
    state.update(status=status, updated_utc=now(), **extra)
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(STATE)
    print(status,extra,flush=True)
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z': return None
        return (p.stat().st_uid,fields[19],(p/'cmdline').read_bytes())
    except FileNotFoundError:return None
def verify_sources():
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source changed after promotion plan: '+name)
def start(script, log, env):
    args=[str(ROOT/'tools/miniconda3/envs/GMT/bin/python'),str(ROOT/'tools'/script)]
    with (ROOT/'logs'/log).open('xb') as output:
        child=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    with (ROOT/'reports/commands_used.sh').open('a') as f:f.write('\n# Runtime promotion controller\n'+shlex.join(args)+'\n')
    return child
try:
    current=identity(plan['diagnostic_pid'])
    if current is not None:
        assert current[0]==os.getuid() and current[1]==plan['diagnostic_start_ticks']
        assert str(ROOT/'tools/diagnose_native_backward.py').encode() in current[2]
    save('waiting_for_exact_diagnostic')
    while current is not None:
        if current[1]!=plan['diagnostic_start_ticks']:raise RuntimeError('Diagnostic PID identity changed')
        time.sleep(20)
        current=identity(plan['diagnostic_pid'])
    output=ROOT/'outputs/diagnose_native_backward_v11_four_gpu_batch4'
    log=(ROOT/'logs/diagnose_native_backward_v11_four_gpu_batch4.log').read_text()
    if 'DIAGNOSTIC_100_UPDATES_COMPLETED_NOT_FORMAL' not in log:
        raise RuntimeError('Diagnostic ended without 100-update completion marker; no restart')
    if 'Traceback (most recent call last)' in log:
        raise RuntimeError('Diagnostic traceback present; inspect before promotion')
    rows=[json.loads(line) for line in (output/'metrics.json').read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get('iteration')!=100:raise RuntimeError('Diagnostic metrics do not reach100')
    for row in rows:
        for key,value in row.items():
            if 'loss' in key and isinstance(value,(float,int)) and not math.isfinite(value):raise RuntimeError('Nonfinite diagnostic loss')
    if not any('total_loss' in row for row in rows):raise RuntimeError('Missing diagnostic losses')
    for rank in range(4):
        last=json.loads((output/'runtime'/f'progress_rank{rank}.jsonl').read_text().splitlines()[-1])
        if last['iteration']!=100 or last['event']!='forward_done':raise RuntimeError('Incomplete rank progress')
        if identity(last['pid']) is not None:raise RuntimeError('Diagnostic worker still live')
    verify_sources()
    # Test this four-rank communication layout only after the diagnostic has ended.
    comm_env=os.environ.copy();comm_env.update(plan['selected_environment'])
    comm_env['GMT_COMM_RESULT']=str(ROOT/'manifests/distributed_large_gloo_four_gpu.json')
    comm_args=[str(ROOT/'tools/run_gmt.sh'),str(ROOT/'tools/distributed_large_smoke.py')]
    with (ROOT/'logs/distributed_large_gloo_four_gpu.log').open('xb') as comm_log:
        completed=subprocess.run(comm_args,cwd=ROOT,env=comm_env,stdin=subprocess.DEVNULL,stdout=comm_log,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Four-rank communication gate failed; no formal launch')
    comm_proof=json.loads((ROOT/'manifests/distributed_large_gloo_four_gpu.json').read_text())
    if comm_proof.get('status')!='PASS' or comm_proof.get('world_size')!=4:raise RuntimeError('Four-rank communication proof missing')
    result={'status':'PASS','completed_iterations':100,'loss_finiteness':'PASS','formal_training':False,'verified_utc':now(),'job':plan['diagnostic_job'],'metrics_sha256':hashlib.sha256((output/'metrics.json').read_bytes()).hexdigest()}
    (ROOT/'manifests/native_backward_diagnostic_v11_four_gpu_batch4_result.json').write_text(json.dumps(result,indent=2))
    # Refuse any independently launched project training process.
    subprocess.run([str(ROOT/'tools/miniconda3/envs/GMT/bin/python'),str(ROOT/'tools/check_training_launch.py'),str(os.getpid())],check=True,cwd=ROOT)
    if any((ROOT/f'manifests/stage{s}_complete.json').exists() for s in (1,2)):raise RuntimeError('Formal completion artifact already exists')
    failure=json.loads((ROOT/'manifests/stage1_formal_failure.json').read_text())
    if failure['status']!='FAILED' or failure['formal_checkpoint_created']:raise RuntimeError('Unexpected prior formal state')
    previous=ROOT/'outputs/stage1'
    if not previous.is_dir() or any(previous.glob('*.pth')):raise RuntimeError('Unexpected prior Stage1 output')
    previous_rows=[json.loads(line) for line in (previous/'metrics.json').read_text().splitlines() if line.strip()]
    if not previous_rows or previous_rows[-1]['iteration']!=60:raise RuntimeError('Prior output differs from documented failed attempt')
    if (ROOT/'outputs/stage2').exists() and any((ROOT/'outputs/stage2').iterdir()):raise RuntimeError('Stage2 output already exists')
    if (ROOT/'manifests/runtime_selection.pending').exists():raise RuntimeError('Separate runtime gate pending')
    backup=ROOT/'backup/formal_before_four_gpu';backup.mkdir()
    for name in ['manifests/selected_runtime.env','manifests/selected_training_runtime.json','manifests/training_pipeline_state.json','manifests/finish_pipeline_state.json']:
        source=ROOT/name
        if source.exists():
            target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    previous.rename(backup/'stage1')
    for name in ['logs/train_stage1.log','logs/train_stage2.log','logs/smoke_stage1.log','logs/smoke_stage2.log','logs/finish_pipeline.log']:
        source=ROOT/name
        if source.exists():
            target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    flags=plan['selected_environment']
    runtime=ROOT/'manifests/selected_runtime.env';temporary=runtime.with_suffix('.tmp')
    temporary.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in flags.items()));temporary.replace(runtime)
    snapshot=ROOT/'reports/selected_runtime_four_gpu';snapshot.mkdir()
    files={}
    for name in list(plan['source_sha256'])+['manifests/selected_runtime.env']:
        source=ROOT/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);files[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot/'git_diff.patch').write_bytes(subprocess.check_output(['git','-C',str(ROOT/'code/GMT'),'diff','--binary']))
    selected={'selected_utc':now(),'runtime':flags,'world_size':4,'batch':4,'diagnostic_proof':result,'fresh_smoke_required':True,'files':files,'formal_iterations_each':20000,'scope':'CPU post-backward averaging; explicit runtime deviation; full-stage stability not yet proven'}
    (ROOT/'manifests/selected_training_runtime.json').write_text(json.dumps(selected,indent=2))
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('GMT_DIAGNOSTIC_','GMT_NATIVE_')) or key=='GMT_REUSE_STAGE1_SMOKE':env.pop(key)
    env.update(flags);env['GMT_SMOKE_VARIANT']='four_gpu';env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(4)))
    training=start('train_pipeline.py','train_pipeline_four_gpu.log',env)
    save('fresh_smoke_and_formal_pipeline_started',training_controller_pid=training.pid)
    for attempt in range(30):
        if training.poll() is not None:raise RuntimeError('Training controller exited during startup')
        try:training_state=json.loads((ROOT/'manifests/training_pipeline_state.json').read_text())
        except json.JSONDecodeError:training_state={}
        if training_state.get('pid')==training.pid:
            if training_state.get('current','').startswith('FAILED'):raise RuntimeError('Training controller startup failed')
            break
        time.sleep(1)
    else:raise RuntimeError('Training controller state did not become visible; inspect live process, no retry')
    finish=start('finish_pipeline.py','finish_pipeline.log',env)
    save('controllers_started_pending_full_training_and_evaluation',training_controller_pid=training.pid,finish_controller_pid=finish.pid)
except Exception as error:
    save('FAILED_no_automatic_retry',error=str(error))
    raise
