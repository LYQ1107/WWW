"""Reconcile only known post-inference failures, then finish all declared P0 data."""
import time
from jev_phase16_common import *


def main():
    protect();start=time.monotonic();folder=OUT/'P0_completion'
    protocol=read(REPORTS/'PREREGISTRATION.json')
    expected=[OUT/'P0_r2/full'/i['policy']/f"video{i['video']:02d}/RESULT.json"
              for i in protocol['P0']['full_replays']]
    expected += [OUT/'P0_r2/windows'/p/f'video{v:02d}/RESULT.json'
                 for p in ['v1','v2','v3'] for v in TRAIN]
    queue_file=OUT/'P0_queue_r2/RESULT.json'
    while True:
        for video in TRAIN:
            result=OUT/'P0_r2/full/original'/f'video{video:02d}/RESULT.json'
            failed_log=OUT/'P0_queue_r2'/f'full_original_video{video}.log'
            if result.exists() or not failed_log.exists():continue
            text=failed_log.read_text()
            missing=str(XV/'commitment_dataset_v2'/f'video{video:02d}/RAW_PREDICTIONS.json')
            if 'FileNotFoundError' not in text or missing not in text:continue
            log=OUT/f'P0_ORIGINAL{video}_COMPLETION_REPAIR.log'
            with log.open('a') as stream:
                subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase16_original_completion_repair.py'),
                    '--video',str(video)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print('PHASE16_POSTPROCESSING_REPAIR_FINISHED',video,flush=True)
        completed=sum(p.exists() and read(p)['status']=='COMPLETE' for p in expected)
        save(folder/'PROGRESS.json',dict(status='WAITING_FOR_ALL_DECLARED_P0_RESULTS',
            completed=completed,total=len(expected),seconds=time.monotonic()-start))
        if queue_file.exists():
            queue=read(queue_file)
            assert completed==len(expected),('P0 has unreconciled failures',queue.get('failed'))
            assert {x['key'] for x in queue['failed']}=={f'full_original_video{v}' for v in TRAIN}
            assert all(x['returncode']==1 for x in queue['failed'])
            break
        time.sleep(10)
    original=[read(OUT/'P0_r2/full/original'/f'video{v:02d}/RESULT.json') for v in TRAIN]
    repairs=read(REPORTS/'P0_REFERENCE_PATH_REPAIR.json')
    repairs.update(status='ALL_FOUR_COMPLETED_NATIVE_TRACES_RECOVERED_NO_RERUN',
        repaired=[ref(OUT/'P0_r2/full/original'/f'video{v:02d}/RESULT.json') for v in TRAIN],
        recovered_row_counts={str(v):len(read(r['exact_original_predictions']['path'])) for v,r in zip(TRAIN,original)},
        original_queue=ref(queue_file),original_errors_retained=queue['failed'],
        actual_native_actor_source=read(OUT/'source_P0_r2.json'),
        original_manager_status=queue['status'],
        reconciled_scientific_completion='all19 full replays and72 H32 branches complete; four post-inference path errors retained')
    save(REPORTS/'P0_REFERENCE_PATH_REPAIR.json',repairs)
    save(folder/'PROGRESS.json',dict(status='AGGREGATING_COMPLETE_P0',completed=completed,total=len(expected)))
    with (OUT/'P0_AGGREGATION.log').open('w') as stream:
        subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase16_recoverability.py')],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    gate=read(REPORTS/'PHASE16_P0_GO_NO_GO.json')
    save(folder/'RESULT.json',dict(status='COMPLETE',binding=binding(inputs=[ref(queue_file)],
        evaluator='all declared run reconciliation and frozen P0 aggregation',
        scope='P0 only; no automatic prototype implementation or training'),
        original_dispatcher_status=queue['status'],all_results_present=True,
        no_GPU_or_optimizer_rerun=True,P0_gate=ref(REPORTS/'PHASE16_P0_GO_NO_GO.json')))
    protect()
    subprocess.run(['git','add','reports/JEV_PHASE16','docs/JEV_PHASE16_EVIDENCE_RECOVERY.md'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete all Phase XVI P0 native history audits and retain four aggregation failures'],cwd=ROOT,check=True)
    ssh=subprocess.check_output(['git','config','--get','core.sshCommand'],cwd=ROOT,text=True).strip()
    ssh+=' -o ConnectTimeout=20 -o ConnectionAttempts=1'
    subprocess.run(['git','-c','core.sshCommand='+ssh,'push','origin',BRANCH],cwd=ROOT,check=True,timeout=120)
    save(folder/'PROGRESS.json',dict(status='COMPLETE_PUSHED',gate=gate['status'],completed=completed,total=len(expected)))
    print('PHASE16_ALL_P0_COMPLETE_PUSHED',gate['status'],flush=True)


if __name__=='__main__':main()
