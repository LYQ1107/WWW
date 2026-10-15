"""Conservative quota correction; shorten only GPU residency, keep all updates.

A CPU shadow unexpectedly saved CUDA RNG states. Charge its entire measured
lifetime plus startup margin on all ten devices, although no GPU model ran.
Supersede the old1.4h GPU residence by<=1.32h, then exact-state CPU continuation.
Original training code, data, optimizer, steps and endpoints remain unchanged.
"""
import argparse,json,os,signal,subprocess,time,datetime
from pathlib import Path
from pirga_indexed_store import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--allocation-sha',required=True);a=p.parse_args()
    if sha(a.allocation)!=a.allocation_sha:raise ValueError('Unpinned corrective allocation')
    allocation=json.loads(a.allocation.read_text());runtime=a.root.parent;protocol=runtime/'PHASE1_MATCHED_PROTOCOL.json';fit=a.root/'fit';statusfile=a.root/'CORRECTIVE_RESOURCE_LEDGER.json'
    if allocation['parent_aggregate_worst_case_GPU_hours']>4:raise ValueError('Parent quota overrun')
    if sha(protocol)!=allocation['training_protocol_sha256']:raise ValueError('Frozen protocol changed')
    state=dict(status='WATCHING_OWN_GPU_RESIDENCY',pid=os.getpid(),allocation_sha256=a.allocation_sha,old_ledger_reset=False,additional_GPU_grant=False,scientific_schedule_changed=False)
    def save():
        state['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=statusfile.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(statusfile)
    try:
        save()
        while True:
            old=json.loads((a.root/'RESOURCE_LEDGER.json').read_text());state['original_GPU_wall_hours']=old['GPU_wall_hours'];save()
            if old['status']=='COMPLETE_FIXED5EPOCHS':state.update(status='COMPLETE_ON_GPU_BEFORE_CORRECTED_CAP');save();return
            if old['status']!='RUNNING_GPU':raise RuntimeError('Unexpected old supervisor state before correction')
            if old['GPU_wall_hours']>=allocation['GPU_stop_trigger_hours']:
                pid=old['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                if 'train_pirga_grouped_phase1.py' not in cmd or str(fit) not in cmd or '--device cuda' not in cmd:raise ValueError('Refuse signaling unrelated process')
                os.killpg(pid,signal.SIGTERM);state.update(status='REQUESTED_EXACT_COMPLETED_STEP_CHECKPOINT',GPU_pid=pid);save();break
            time.sleep(2)
        start=time.monotonic()
        while True:
            old=json.loads((a.root/'RESOURCE_LEDGER.json').read_text())
            if old['status']=='FAILED_RETAIN_EVIDENCE' and not Path(f'/proc/{pid}').exists():break
            if time.monotonic()-start>45:raise RuntimeError('Original supervisor did not terminate safely')
            time.sleep(1)
        s=json.loads((fit/'STATUS.json').read_text())
        if s['status']!='RESOURCE_CHECKPOINT_READY_FOR_CPU_CONTINUATION':raise RuntimeError('No exact checkpoint for CPU')
        state.update(status='RUNNING_CPU_CONTINUATION',GPU_wall_hours=old['GPU_wall_hours'],resume_sha256=sha(fit/'RESUME.pth'),
            original_supervisor_error_is_expected_corrective_handoff=True)
        if old['GPU_wall_hours']>allocation['GPU_residency_maximum_hours']:raise RuntimeError('Corrective GPU residency breached')
        command=[allocation['python'],'-u',str(Path(__file__).with_name('train_pirga_grouped_phase1.py')),'--protocol',str(protocol),
            '--protocol-sha',allocation['training_protocol_sha256'],'--out',str(fit),'--device','cpu','--resume',str(fit/'RESUME.pth')]
        with (a.root/'CORRECTIVE_CPU_CONTINUATION.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),start_new_session=True)
            state['CPU_pid']=child.pid;save();code=child.wait()
        if code:raise RuntimeError('Corrective CPU continuation failed')
        if json.loads((fit/'STATUS.json').read_text())['status']!='COMPLETE':raise RuntimeError('Incomplete CPU fit')
        state.update(status='COMPLETE_FIXED5EPOCHS',CPU_pid=None,scientific_success=False);save()
    except BaseException as e:
        state.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));save();raise

if __name__=='__main__':main()
