from dataclasses import replace
import numpy as np
import pytest
from worldobject_synthetic import core,obs,gid
from worldmot.tracking.world_object_config import WorldObjectConfig
from worldmot.tracking.joint_world_object_assignment import JointWorldObjectAssignment
from worldmot.models.world_object_model_inputs import model_inputs


def test_g1_g2_g3_g4_are_independent_not_one_use_geometry_alias():
    c=WorldObjectConfig(3,enable_sync_geometry=False,enable_temporal_geometry=True,
                        enable_spatial_state_update=False,enable_head_geometry=True)
    assert not c.sync_geometry and c.temporal_geometry and not c.spatial_state_update and c.head_geometry
    bank=core(enable_spatial_state_update=False);bank.step([obs()],0)
    assert not bank.bank.objects[1].geometry_valid
    c2=replace(c,enable_head_geometry=False)
    inputs=model_inputs([], [obs(1),obs(2)],c2,0)
    assert inputs[2].shape==(0,2,16) and inputs[5].shape==(2,2,16)
    assert not inputs[5].numpy().any() and c2.temporal_geometry


def test_reliable_different_people_current_geometry_conflict_still_vetoes_fusion():
    c=core(enable_motion_prediction=True,geometry_policy='UNCERTAINTY_SOFT',enable_safe_identity_memory=True)
    r=c.step([obs(1),obs(2,center=(10,0,3))],0)
    assert gid(r,1)!=gid(r,2)


def test_alignment_failed_or_nonfinite_geometry_abstains_without_losing_box():
    c=core(enable_motion_prediction=True,geometry_policy='UNCERTAINTY_SOFT',enable_safe_identity_memory=True)
    c.step([obs()],0)
    r=c.step([obs(center=(np.nan,0,0),geometry=False,timestamp=1)],1)
    assert gid(r,1)==1 and len(r['tracks_2d_by_camera']['1'])==1
    assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==0


def test_spatial_abstain_ignores_history_location_but_keeps_visual_requirement():
    c=core(geometry_policy='SPATIAL_ABSTAIN');c.step([obs()],0)
    assert gid(c.step([obs(center=(100,0,3),timestamp=1)],1),1)==1
    r=c.step([obs(center=(100,0,3),feature=(0,1,0),timestamp=2)],2)
    assert gid(r,1)!=1


def test_geometry_off_does_not_change_head_shape_or_erase_current_detections():
    c=core(enable_sync_geometry=False,enable_temporal_geometry=False,enable_spatial_state_update=False)
    r=c.step([obs(1),obs(2,center=(100,0,3))],0)
    assert gid(r,1)==gid(r,2)
    assert c.config.head_geometry


def test_invalid_motion_frame_transform_fails_explicitly():
    from worldmot.tracking.world_object_motion_state import PersistentWorldMotionState as Motion
    s=Motion.initialize([0,0,0],np.eye(3),0)
    with pytest.raises(ValueError,match='SIM3'):s.transform(-1,np.eye(3),np.zeros(3))
