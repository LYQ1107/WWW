"""Fixed assignments, atomic progress and resume for sealed native runs."""
import argparse,os,subprocess,time,json
from jev_phase10_common import *


def main(phase):
 protect();source=binding();assert not source['dirty']
 if phase=='heldout':protocol=heldout_guard(HELDOUT[0]);videos=HELDOUT
 else:protocol=json.loads((REPORTS/'PHASE10_VALIDATION_CONTROLLER_PROTOCOL.json').read_text());videos=VAL
 slots=[4,5,6,7,1,9,2,3,8,0];queues={g:[]for g in slots};tasks=[(case['name'],video)for case in protocol['cases']for video in videos]
 for index,task in enumerate(tasks):queues[slots[index%len(slots)]].append(task)
 running={};completed=[];failed=[];logs=OUT/'execution_logs';logs.mkdir(exist_ok=True)
 while any(queues.values())or running:
  for gpu in slots:
   if gpu in running or not queues[gpu]or failed:continue
   name,video=queues[gpu].pop(0);log=logs/f'{phase}_{name}_video{video:02d}.log';handle=log.open('a');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');p=subprocess.Popen([PYTHON,'reproduction_tools/run_jev_phase10_closed_loop.py','--phase',phase,'--case',name,'--video',str(video)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);running[gpu]={'process':p,'handle':handle,'case':name,'video':video,'GPU':gpu,'log':str(log)};print('CLOSED_LOOP_START',phase,name,video,gpu,flush=True)
  for gpu,d in list(running.items()):
   code=d['process'].poll()
   if code is None:continue
   d['handle'].close();item={k:v for k,v in d.items()if k not in ['process','handle']}|{'exit_code':code};(completed if code==0 else failed).append(item);del running[gpu];print('CLOSED_LOOP_FINISH',phase,item,flush=True)
  save(OUT/f'{phase.upper()}_EXECUTION.json',{'status':'FAIL'if failed else'RUNNING','phase':phase,'source_commit':source['source_commit'],'pending':{str(g):q for g,q in queues.items()},'running':[{k:v for k,v in d.items()if k not in ['process','handle']}|{'PID':d['process'].pid}for d in running.values()],'completed':completed,'failed':failed})
  if failed and not running:break
  if any(queues.values())or running:time.sleep(5)
 assert not failed,failed
 save(OUT/f'{phase.upper()}_COMPLETE.json',{'status':'COMPLETE','phase':phase,'source_commit':source['source_commit'],'protocol_SHA256':sha(REPORTS/('PHASE10_CHECKPOINT_AND_HELDOUT_FREEZE.json'if phase=='heldout'else'PHASE10_VALIDATION_CONTROLLER_PROTOCOL.json')),'all_runs_complete':True,'runs':len(tasks),'cases':len(protocol['cases']),'videos':videos});print('ALL_NATIVE_CLOSED_LOOPS_COMPLETE',phase,len(tasks),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','heldout'],required=True);main(p.parse_args().phase)
