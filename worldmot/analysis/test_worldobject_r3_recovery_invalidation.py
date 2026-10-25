"""R3-only regression tests. Frozen R2 keeps its original behavior."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from worldobject_synthetic import core,obs,gid
from test_worldobject_memory import evidence


def learned(m,n):
    result=evidence(m,n);result['assignment_logits'][:]=8.;result['null_logits'][:]=-2.
    result['birth_logits'][:]=-10.;result['birth_pair_logits']=np.full((n,n),10.)
    return result


def test_recovery_invalidates_stale_position_then_reinitializes_from_current_geometry():
    for method in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3'):
        tracker=core(method)
        tracker.step([obs(1),obs(2)],0,evidence=learned(0,2));tracker.step([],1,evidence=learned(1,0))
        old=tracker.bank.objects[1].center.copy();embedding=tracker.bank.objects[1].identity_embedding.copy()
        recovered=tracker.step([obs(1,center=(20,0,3),timestamp=2),obs(2,center=(20,0,3),timestamp=2)],2,evidence=learned(1,2))
        assert gid(recovered,1)==gid(recovered,2)==1
        q=tracker.bank.objects[1];assert not q.geometry_valid and q.missed==0
        assert q.support_points.shape==(0,3)
        np.testing.assert_array_equal(q.center,old)
        np.testing.assert_array_equal(q.identity_embedding,embedding)
        assert recovered['tracks_3d'][0]['center'] is None
        following=tracker.step([obs(1,center=(20,0,3),timestamp=3),obs(2,center=(20,0,3),timestamp=3)],3,evidence=learned(1,2))
        assert gid(following,1)==gid(following,2)==1
        assert q.geometry_valid
        np.testing.assert_allclose(q.center,[20,0,3])


def test_ordinary_reliable_history_update_keeps_valid_spatial_state():
    tracker=core();tracker.step([obs()],0,evidence=learned(0,1))
    result=tracker.step([obs(center=(.1,0,3),timestamp=1)],1,evidence=learned(1,1))
    assert gid(result,1)==1 and tracker.bank.objects[1].geometry_valid


def test_rejected_geometry_is_not_inserted_when_reappearance_has_no_valid_followup():
    tracker=core();tracker.step([obs()],0,evidence=learned(0,1));tracker.step([],1,evidence=learned(1,0))
    tracker.step([obs(2,center=(20,0,3),timestamp=2)],2,evidence=learned(1,1))
    tracker.step([obs(2,center=(999,0,3),timestamp=3,geometry=False)],3,evidence=learned(1,1))
    assert not tracker.bank.objects[1].geometry_valid
    assert tracker.bank.last_assignments[(2,0)]==1
