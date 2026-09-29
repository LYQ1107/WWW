"""Record GPU/process state and refuse duplicate project training launches."""
from pathlib import Path
import os,sys,subprocess,datetime,json
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');excluded={int(sys.argv[1]),os.getpid()};duplicates=[]
with (r/'logs/gpu_launch_checks.log').open('a') as f:
 f.write('\nUTC '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
 subprocess.run(['nvidia-smi'],stdout=f,stderr=subprocess.STDOUT,check=True)
 # Inspect exact argv and cwd to avoid matching shell source strings or unrelated jobs.
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in excluded:continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if not any(Path(a).name in ('run_training_smoke.py','train_net.py','run_observed_training.py','diagnose_native_backward.py','diagnose_single_visible.py') for a in args):continue
   if (p/'cwd').resolve()!=r/'code/GMT':continue
   duplicates.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 f.write('Other active project training PIDs: '+json.dumps(duplicates)+'\n')
if duplicates:raise RuntimeError('Existing project training process; refusing duplicate launch: '+str(duplicates))
