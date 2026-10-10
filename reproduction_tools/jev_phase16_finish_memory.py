"""Visible sequential handoff; no dependency stage starts before native queue exit."""
from jev_phase16_common import *
import time


def main():
    begin=time.monotonic();folder=OUT/'memory_completion';folder.mkdir(parents=True,exist_ok=True)
    while not (OUT/'P2_P3_queue/RESULT.json').exists():
        p=OUT/'P2_P3_queue/PROGRESS.json'
        save(folder/'PROGRESS.json',dict(status='WAITING_NATIVE_QUEUE',queue=read(p) if p.exists() else None,seconds=time.monotonic()-begin))
        time.sleep(15)
    assert read(OUT/'P2_P3_queue/RESULT.json')['status']=='COMPLETE'
    save(folder/'PROGRESS.json',dict(status='AGGREGATING_NATIVE_MEMORY',seconds=time.monotonic()-begin))
    subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase16_memory_results.py')],cwd=ROOT,check=True)
    selected=['MATCH_VS_MEMORY_CAUSAL_FACTORIAL','NATIVE_FUTURE_SAFETY_GO_NO_GO','RECOVERY_AND_CONTAMINATION_EFFECTS','NATIVE_MEMORY_PARITY','LABEL_ELIGIBILITY','SAFE_MEMORY_ADMISSION']
    subprocess.run(['git','add']+[f'reports/JEV_PHASE16/{n}.json' for n in selected],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete 96 real mutated futures and frozen admission safety/label gates'],cwd=ROOT,check=True)
    ssh=subprocess.check_output(['git','config','--get','core.sshCommand'],cwd=ROOT,text=True).strip()+' -o ConnectTimeout=20 -o ConnectionAttempts=1'
    subprocess.run(['git','-c','core.sshCommand='+ssh,'push','origin',BRANCH],cwd=ROOT,check=True,timeout=120)
    save(folder/'PROGRESS.json',dict(status='NATIVE_RESULTS_PUSHED_STARTING_FULL_IMAGES',seconds=time.monotonic()-begin))
    subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase16_p8_queue.py')],cwd=ROOT,check=True)
    save(folder/'PROGRESS.json',dict(status='FULL_IMAGE_MATLAB_AND_LATENCY_COMPLETE',seconds=time.monotonic()-begin))


if __name__=='__main__':main()
