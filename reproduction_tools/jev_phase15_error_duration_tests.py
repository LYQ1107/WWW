"""Identity errors persist through mixing; bootstrap owners and gaps matter."""
from jev_phase15_common import *
from jev_phase15_evaluate import owner_propagation


def main():
    trace=[dict(key=[frame,0],GT_OFFLINE_ONLY=gt,identity=identity) for frame,gt,identity in
        [(0,2,10),(1,2,10),(2,2,20),(4,2,10),(6,2,10),(7,None,10)]]
    result=owner_propagation(trace,{10:1})
    assert result['counts']['wrong_fixed_owner_observations']==4
    episodes=result['episodes'];assert [e['observed_frames'] for e in episodes]==[2,1,1]
    assert [e['right_censored'] for e in episodes]==[False,True,True]
    assert episodes[0]['end']==1 and episodes[1]['end_reason']=='observation gap'
    assert owner_propagation(trace)['counts']['wrong_fixed_owner_observations']==0, 'bootstrap omission test lost meaning'
    save(OUT/'error_duration_contract/RESULT.json',dict(status='PASS',binding=binding(seed=20261009),
        bootstrap_owner_used=True,mixed_history_does_not_reset_error=True,unknown_not_correction=True,
        gap_right_censored=True,video_end_right_censored=True,actual_next_frame_correction_distinguished=True))
    print('PHASE15_ERROR_DURATION_TEST_PASS',flush=True)


if __name__=='__main__':main()
