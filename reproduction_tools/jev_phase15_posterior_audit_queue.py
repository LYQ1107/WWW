"""Freeze one-feature-family intervention and all-query coverage audit."""
from jev_phase15_common import *
from jev_phase15_queue import pin,run_queue


def main():
    source=pin('source_posterior_attribution_v1');jobs=[]
    for video in TRAIN:
        jobs.append(dict(key=f'posterior_masked_TRAIN{video}',script='jev_phase15_pilot_native.py',
            args=['--video',str(video),'--version','1','--posterior-input','masked'],
            result=str(OUT/'posterior_masked_native_v1/F_full/pilot'/f'video{video:02d}'/'RESULT.json')))
    jobs.append(dict(key='all_query_coverage_v1',script='jev_phase15_all_query_audit.py',args=['--version','1'],
        result=str(OUT/'all_query_audit_v1/RESULT.json')))
    outcome=run_queue(source,jobs,'posterior_attribution_queue_v1',max_active=5);assert not outcome['failed'],outcome


if __name__=='__main__':main()
