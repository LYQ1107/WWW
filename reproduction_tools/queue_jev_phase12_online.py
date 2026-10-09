"""Run the entire frozen validation matrix; stop new launches on any failure."""
import argparse,os,subprocess,time,json
from pathlib import Path
from jev_phase12_common import OUT,PYTHON,save

def main(source,gpus):
    source=Path(source).resolve();protocol=json.loads((source/'reports/JEV_PHASE12/ONLINE_PROTOCOL.json').read_text());assert protocol['status']=='FROZEN_BEFORE_MOT_VALIDATION'
    pending=[{'case':c['name'],'video':v} for c in protocol['cases'] for v in protocol['videos']];running={};finished=[];failed=[];total=len(pending)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    while pending or running:
        for gpu,(job,process) in list(running.items()):
            code=process.poll()
            if code is None:continue
            done={**job,'exit_code':code};(finished if code==0 else failed).append(done);del running[gpu]
        if failed:pending=[]
        for gpu in gpus:
            if gpu in running or not pending:continue
            job=pending.pop(0);case=job['case'];video=job['video'];path=OUT/'validation_closed_loop_v1'/case/f'video{video:02d}/RESULT.json'
            if path.exists():finished.append({**job,'exit_code':0,'already_complete':True});continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=OUT/f'online_{case}_video{video:02d}.log';handle=log.open('a')
            process=subprocess.Popen([PYTHON,'reproduction_tools/run_jev_phase12_closed_loop.py','--video',str(video),'--case',case],cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);handle.close();job.update(gpu=gpu,pid=process.pid,log=str(log));running[gpu]=(job,process)
            print('ONLINE_LAUNCH',case,video,gpu,process.pid,flush=True)
        status='FAILED' if failed else 'RUNNING' if pending or running else 'COMPLETE'
        save(OUT/'ONLINE_QUEUE_PROGRESS.json',{'status':status,'source':str(source),'source_commit':commit,'pending':pending,'running':[j for j,p in running.values()],'finished':finished,'failed':failed,'total':total,'timestamp_UTC':time.time()})
        if pending or running:time.sleep(2)
    if failed:raise RuntimeError('failed native actors retained; new launches stopped')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--gpus',default='9,0,2,3,4,5,7,8');a=p.parse_args();main(a.source,[int(v) for v in a.gpus.split(',')])
