"""Complete live DEV diagnostics plus true runtime on a fixed LAST pilot."""
import argparse
from jev_phase15_common import *
from jev_phase15_queue import pin,run_queue


def main(version=3):
    source=pin(f'source_complete_live_diagnostic_v{version}');jobs=[]
    for video in DEV:
        jobs.append(dict(key=f'live_F_LAST_v{version}_DEV{video}',script='jev_phase15_evaluate.py',
            args=['--video',str(video),'--version',str(version),'--phase','pilot','--live'],
            result=str(OUT/f'pilot_online_v{version}/F_full_seed20261009/live'/f'video{video:02d}/RESULT.json')))
    jobs.append(dict(key=f'true_runtime_F_v{version}',script='jev_phase15_latency.py',args=['--version',str(version)],
        result=str(OUT/f'latency_v{version}/RESULT.json')))
    r=run_queue(source,jobs,f'complete_live_diagnostic_queue_v{version}',max_active=4);assert not r['failed'],r
    command=[PYTHON,str(source/'reproduction_tools/jev_phase15_matlab.py'),'--version',str(version)]
    subprocess.run(command,cwd=source,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=3);a=p.parse_args();main(a.version)
