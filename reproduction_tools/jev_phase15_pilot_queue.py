"""Finite Tiny/Pilot then real TRAIN safety branches, never automatic 20k."""
import argparse
from jev_phase15_common import *
from jev_phase15_queue import pin,run_queue


def main(version=1):
    protect();assert read(REPORTS/'COMMITMENT_LABEL_AUDIT.json')['status']=='PASS'
    assert read(REPORTS/'NATIVE_STATE_CONTRACT.json')['status']=='PASS_FOR_IMPLEMENTED_NO_ALIAS_ADAPTER'
    assert read(REPORTS/'COMMITMENT_FEASIBILITY_GO_NO_GO.json')['mechanism_GO']
    source=pin(f'source_full_payload_pilot_v{version}')
    for phase in ['tiny','pilot']:
        path=OUT/f'training_full_payload_v{version}'/'F_full/seed20261009'/phase/'RESULT.json'
        jobs=[dict(key=f'F_full_{phase}_v{version}',script='jev_phase15_train.py',
             args=['--arm','F_full','--phase',phase,'--version',str(version)],result=str(path))]
        result=run_queue(source,jobs,f'{phase}_full_payload_queue_v{version}',max_active=1)
        assert not result['failed'],result['failed']
        measurement=read(path);assert measurement['actual_updates']==(128 if phase=='tiny' else 1500)
        # Missing correction audit support is preserved as a failed scientific
        # gate. This diagnostic pilot must never launch formal training.
    jobs=[dict(key=f'F_full_native_v{version}_TRAIN{video}',script='jev_phase15_pilot_native.py',
          args=['--video',str(video),'--version',str(version)],
          result=str(OUT/f'pilot_native_v{version}'/'F_full/pilot'/f'video{video:02d}'/'RESULT.json')) for video in TRAIN]
    result=run_queue(source,jobs,f'pilot_native_queue_v{version}',max_active=4)
    assert not result['failed'],result['failed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);a=p.parse_args();main(a.version)
