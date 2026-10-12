"""Read-only process monitor: no restarts, training, or scientific gate changes."""
from jev_recovery_common import *


def inspect_job(name,job):
    pid=job['PID'];proc=pathlib.Path('/proc',str(pid))
    value=dict(job=name,PID=pid,GPU=job.get('GPU'),registry_status=job['status'])
    try:
        fields=(proc/'stat').read_text().split(') ',1)[1].split()
        command=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
        value.update(process_state=fields[0],parent_PID=int(fields[1]),CPU_ticks=int(fields[11])+int(fields[12]),
                     start_ticks=int(fields[19]),actual_command=command,
                     command_matches_registry=command==job['command'],
                     actual_children=(proc/'task'/str(pid)/'children').read_text().split())
        if fields[0]=='Z':value['process_observation']='EXITED_AWAITING_SUPERVISOR_REAP'
        elif not value['command_matches_registry']:value['process_observation']='PID_COMMAND_MISMATCH_REQUIRES_REVIEW'
        else:value['process_observation']='VERIFIED_LIVE_PROCESS'
    except FileNotFoundError:value['process_observation']='PROCESS_HANDLE_MISSING_RECHECK_SUPERVISOR'
    log=pathlib.Path(job['log'] if isinstance(job['log'],str) else job['log']['path'])
    if log.exists():value.update(log_bytes=log.stat().st_size,log_mtime_ns=log.stat().st_mtime_ns)
    return value


def main():
    protect();prior={};last_signature=None
    while True:
        registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');live=[]
        for name,j in registry['jobs'].items():
            if j['status']=='RUNNING':
                row=inspect_job(name,j);old=prior.get(name,{})
                row['CPU_or_log_progress_since_last_check']=any(row.get(k)!=old.get(k) for k in ['CPU_ticks','log_bytes','log_mtime_ns'])
                live.append(row);prior[name]=row
        phases={}
        for label,path in [('progression',OUT/'progression/PROGRESS.json'),('finish',OUT/'finish/READY.json'),
                           ('supplementary_fusion',REPORTS/'DIAGNOSTIC_FUSION_RESULT_INDEX.json'),
                           ('frozen_online',REPORTS/'QUESTION_INTERVENTION_NATIVE_INDEX.json')]:
            if path.exists():phases[label]=read(path).get('status')
        failed=[ref(p) for p in unreviewed_failures()]
        status=dict(status='VERIFIED_MONITORING',updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    live_jobs=live,phases=phases,unreviewed_failures=failed,
                    free_GiB={p:shutil.disk_usage(p).free/2**30 for p in ['/home','/data1']},
                    automatic_restart=False,scientific_success_not_inferred=True)
        save(OUT/'live_monitor/STATUS.json',status)
        signature=([(r['job'],r['process_observation']) for r in live],phases,[f['path'] for f in failed])
        if signature!=last_signature:
            save(OUT/'live_monitor/events'/f'{time.time_ns()}.json',status)
            print('MONITOR_CHANGE',status['updated_UTC'],signature,flush=True);last_signature=signature
        if phases.get('progression')=='BOUNDED_TRAIN_ONLINE_ROUTES_COMPLETE' and 'finish' in phases:
            save(OUT/'live_monitor/ORIGINAL_QUEUE_TERMINAL_OBSERVATION.json',status)
            print('ORIGINAL_QUEUE_FINISHED_REQUIRES_FULL_EVIDENCE_REVIEW',flush=True)
            break
        time.sleep(30)


if __name__=='__main__':main()
