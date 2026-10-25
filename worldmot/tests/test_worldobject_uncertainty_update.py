import numpy as np
import pytest
from worldmot.tracking.world_object_motion_state import (PersistentWorldMotionState as Motion,
    conservative_measurement,measurement_covariance)
from worldobject_synthetic import obs


def test_kalman_numerical_update_and_joseph_positive_definiteness():
    s=Motion.initialize([0,0,0],np.eye(3),0)
    assert s.update([2,0,0],np.eye(3),0)
    np.testing.assert_allclose(s.position,[1,0,0]);np.testing.assert_allclose(s.covariance[:3,:3],np.eye(3)*.5)
    assert np.linalg.eigvalsh(s.covariance).min()>0


@pytest.mark.parametrize('position',[[np.nan,0,0],[np.inf,0,0]])
def test_nonfinite_measurement_abstains_without_poisoning(position):
    s=Motion.initialize([0,0,0],np.eye(3),0)
    assert not s.update(position,np.eye(3),1)
    assert np.isfinite(s.state).all() and np.isfinite(s.covariance).all()
    assert s.last_trusted_geometry_timestamp==0


def test_non_psd_covariance_rejected_but_roundoff_regularized():
    with pytest.raises(ValueError,match='POSITIVE'):Motion.initialize([0,0,0],np.diag([-1,1,1]),0)
    s=Motion.initialize([0,0,0],np.eye(3),0)
    assert not s.update([0,0,0],np.diag([-1,1,1]),0)
    assert np.linalg.eigvalsh(s.covariance).min()>0


def test_correlated_equal_views_do_not_halve_measurement_uncertainty():
    a,b=obs(1),obs(2)
    center,cov,reliability=conservative_measurement([a,b])
    np.testing.assert_allclose(center,a.world_center)
    np.testing.assert_allclose(cov,measurement_covariance(a)[0]);assert reliability==1


def test_disagreeing_views_inflate_covariance_and_reduce_reliability():
    _,equal,r1=conservative_measurement([obs(1),obs(2)])
    _,conflict,r2=conservative_measurement([obs(1),obs(2,center=(1,0,3))])
    assert conflict[0,0]>equal[0,0] and r2<r1*.1


def test_long_geometry_gap_expands_uncertainty_without_fake_update():
    s=Motion.initialize([0,0,0],np.eye(3),0);s.state[3]=.1
    initial=np.trace(s.covariance);s.predict(100)
    assert np.trace(s.covariance)>initial*100 and s.position[0]==10
    assert s.last_trusted_geometry_timestamp==0


def test_alignment_quality_and_residual_inflate_noise_without_fake_independence():
    a=obs();base,_=measurement_covariance(a)
    a.source={'alignment_confidence':.5,'alignment_residual_p95':.1}
    inflated,point_quality=measurement_covariance(a)
    np.testing.assert_allclose(inflated,(base+np.eye(3)*.01)/.5)
    assert point_quality==1.
    a.source['alignment_confidence']=0
    with pytest.raises(ValueError,match='ALIGNMENT'):measurement_covariance(a)


def test_corrupt_prior_covariance_abstains_then_reinitializes_from_trusted_current():
    from worldobject_synthetic import core,gid
    c=core(enable_motion_prediction=True,enable_safe_identity_memory=True,geometry_policy='UNCERTAINTY_SOFT')
    c.step([obs()],0)
    c.bank.objects[1].spatial_state.covariance[0,0]=-1.
    assert np.linalg.eigvalsh(c.bank.objects[1].spatial_state.covariance).min()<0
    result=c.step([obs(timestamp=1)],1)
    assert gid(result,1)==1 and c.bank.objects[1].geometry_valid
    assert np.linalg.eigvalsh(c.bank.objects[1].spatial_state.covariance).min()>0
    assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==1
