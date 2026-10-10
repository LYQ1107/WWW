"""Read-only final memory and exact partial-overlap integrity, no reader selection."""
from jev_phase16_common import *


def main():
    protect();cases=[];weights=[]
    for video in TRAIN+DEV:
        path=OUT/'P1_r2'/f'video{video:02d}/RESULT.json'
        if not path.exists():continue
        r=read(path);p0=OUT/'P0_r2/full/v3'/f'video{video:02d}/RESULT.json'
        expected=read(p0)['cases'][0]['final_observed_identity_memory_SHA256']
        assert r['status']=='COMPLETE' and r['final_original_identity_and_commitment_memory_SHA256']==expected
        assert r['every_original_logit_and_committed_ID_exact_vs_P0']
        for ck in r['binding']['checkpoints']:
            if ck not in weights:weights.append(ck)
        cases.append(dict(video=video,completed_trial=ref(path),original_native_audit=ref(p0),
            exact_previous_partial_records=r['engineering_exact_overlap_verified'],
            final_native_identity_and_commitment_SHA256=expected,
            every_original_logit_and_ID_exact=True))
    save(REPORTS/'P1_ENGINEERING_EQUIVALENCE.json',dict(status='PASS_ALL7' if len(cases)==7 else 'PARTIAL_PASS_PENDING_COMPLETION',
        binding=binding(checkpoints=weights,inputs=[c['completed_trial'] for c in cases],
            evaluator='exact native logits/IDs/final memory and preserved partial shadow records',
            scope='engineering equivalence only; no retrieval or native benefit gate'),
        completed=len(cases),declared=7,cases=cases,reader_selection_performed=False))
    print('P1_ENGINEERING_EQUIVALENCE',len(cases),sum(c['exact_previous_partial_records'] for c in cases),flush=True)


if __name__=='__main__':main()
