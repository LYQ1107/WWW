import numpy as np
import pytest
from worldmot.tracking.world_object_motion_state import PersistentWorldMotionState as Motion
from worldobject_synthetic import core,obs,gid


def test_cv_analytic_mean_and_white_acceleration_covariance():
    s=Motion.initialize([0,0,0],np.eye(3),0,acceleration_ratio=.3)
    s.state[3:]=[1,2,3];s.predict(2)
    np.testing.assert_allclose(s.position,[2,4,6])
    np.testing.assert_allclose(s.covariance[:3,:3],np.eye(3)*(5+.3*8/3))
    np.testing.assert_allclose(s.covariance[:3,3:],np.eye(3)*(2+.3*4/2))
    np.testing.assert_allclose(s.covariance[3:,3:],np.eye(3)*(1+.3*2))


def test_prediction_is_idempotent_and_rejects_time_reversal():
    s=Motion.initialize([0,0,0],np.eye(3),0);s.predict(3)
    before=s.covariance.copy();s.predict(3);np.testing.assert_array_equal(before,s.covariance)
    with pytest.raises(ValueError,match='NONCAUSAL'):s.predict(2)


def test_verified_fps_conversion_and_explicit_tick_units():
    s=Motion.initialize([0,0,0],np.eye(3),0,ticks_per_second=30)
    s.state[3:]=[2,0,0];s.predict(30);assert s.position[0]==2 and s.time_unit=='seconds'
    assert Motion.initialize([0,0,0],np.eye(3),0).time_unit=='frame_id_tick'


def test_moving_person_keeps_id_and_learns_velocity():
    c=core(enable_motion_prediction=True)
    for t in range(20):
        r=c.step([obs(center=(t*.03,0,3),timestamp=t)],t)
        assert gid(r,1)==1
    state=c.bank.objects[1].spatial_state
    assert abs(state.velocity[0]-.03)<.005
    assert abs(state.position[0]-.57)<.01


def test_prediction_before_head_does_not_leak_future_observations():
    c=core(enable_motion_prediction=True);c.step([obs()],0)
    before=c.bank.objects[1].spatial_state.state.copy()
    with pytest.raises(ValueError,match='FUTURE'):c.step([obs(timestamp=2)],1)
    np.testing.assert_array_equal(c.bank.objects[1].spatial_state.state,before)
    assert c.bank.objects[1].spatial_state.last_prediction_timestamp==0


def test_sim3_transforms_velocity_covariance_and_process_noise():
    s=Motion.initialize([1,2,3],np.eye(3),0);s.state[3:]=[1,0,0]
    r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    s.transform(2,r,[3,0,0],new_frame_id='scaled')
    np.testing.assert_allclose(s.position,[-1,2,6]);np.testing.assert_allclose(s.velocity,[0,2,0])
    np.testing.assert_allclose(s.covariance,np.eye(6)*4)
    np.testing.assert_allclose(s.acceleration_density,np.eye(3)*.04)
    assert not s.update([0,0,0],np.eye(3),1,frame_id='wrong_frame')


def test_scene_reset_discards_motion_and_uses_only_scene_origin():
    c=core(enable_motion_prediction=True);c.step([obs()],0);c.step([obs(timestamp=10)],10)
    c.step([obs(timestamp=0,center=(50,0,3))],0,scene_id=2)
    assert len(c.bank.objects)==1 and c.bank.objects[1].spatial_state.last_prediction_timestamp==0
    np.testing.assert_allclose(c.bank.objects[1].center,[50,0,3])


def test_numeric_overflow_abstains_without_writing_inf_state():
    s=Motion.initialize([0,0,0],np.eye(3),0)
    s.predict(1e200)
    assert not s.motion_valid and np.isfinite(s.state).all() and np.isfinite(s.covariance).all()
    assert s.uncertainty_status=='INVALID_PREDICTION_ABSTAIN'
