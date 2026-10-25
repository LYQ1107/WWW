import numpy as np
import pytest
from worldobject_synthetic import core,obs,gid
from worldmot.tracking.world_object_identity_memory import WorldObjectIdentityMemory


def safe_core(**kwargs):
    return core(enable_motion_prediction=True,geometry_policy='UNCERTAINTY_SOFT',
                enable_safe_identity_memory=True,**kwargs)


def test_recovered_identity_survives_next_frame_spatial_staleness():
    c=safe_core();c.step([obs()],0);c.step([],1)
    a=c.step([obs(2,center=(100,0,3),timestamp=2)],2)
    assert gid(a,2)==1
    assert not c.bank.objects[1].geometry_valid
    b=c.step([obs(2,center=(100.01,0,3),timestamp=3)],3)
    assert gid(b,2)==1 and c.bank.objects[1].geometry_valid
    assert c.bank.objects[1].center[0]>99


@pytest.mark.parametrize('coordinate_views',[False,True])
def test_geometry_failure_preserves_identity_but_does_not_write_bad_position(coordinate_views):
    c=safe_core(enable_multiview_coordination=coordinate_views);c.step([obs()],0)
    old=c.bank.objects[1].center.copy();f=c.bank.objects[1].identity_embedding.copy()
    r=c.step([obs(center=(999,0,3),geometry=False,timestamp=1)],1)
    assert gid(r,1)==1
    np.testing.assert_allclose(c.bank.objects[1].center,old)
    np.testing.assert_array_equal(c.bank.objects[1].identity_embedding,f)
    assert c.bank.objects[1].identity_state.identity_update_status=='SHORT_TERM_ONLY_SPATIAL_UNCERTAIN'


@pytest.mark.parametrize('coordinate_views',[False,True])
def test_spatially_close_but_different_visual_identity_does_not_merge(coordinate_views):
    c=safe_core(enable_multiview_coordination=coordinate_views);c.step([obs()],0)
    r=c.step([obs(feature=(0,1,0),timestamp=1)],1)
    assert gid(r,1)!=1
    np.testing.assert_array_equal(c.bank.objects[1].identity_embedding,[1,0,0])


@pytest.mark.parametrize('coordinate_views',[False,True])
def test_similar_people_compete_without_contaminating_memory(coordinate_views):
    c=safe_core(enable_multiview_coordination=coordinate_views);r=c.step([obs(1,0),obs(1,1,feature=(1,.01,0),center=(10,0,3))],0)
    first=c.bank.objects[1].identity_embedding.copy()
    r=c.step([obs(1,0,timestamp=1),obs(1,1,feature=(1,.01,0),center=(10,0,3),timestamp=1)],1)
    assert len({x['global_id'] for x in r['tracks_2d_by_camera']['1']})==2
    np.testing.assert_array_equal(c.bank.objects[1].identity_embedding,first)
    assert c.bank.objects[1].identity_state.identity_update_status=='IDENTITY_UNCERTAIN_NO_LONG_TERM_UPDATE'


def test_untrusted_or_contrary_view_does_not_overwrite_long_memory():
    m=WorldObjectIdentityMemory(np.array([1.,0,0]))
    m.update([obs(feature=(0,1,0))],1,trusted=False,spatial_trusted=True,dimension=3)
    np.testing.assert_array_equal(m.long_term_embedding,[1,0,0]);assert not m.short_term_embedding_by_camera
    m.update([obs(1),obs(2,feature=(0,1,0))],2,trusted=True,spatial_trusted=True,dimension=3,coordinate_views=True)
    np.testing.assert_allclose(m.long_term_embedding,[1,0,0]);assert 1 in m.short_term_embedding_by_camera
    assert 2 not in m.short_term_embedding_by_camera


def test_visual_only_birth_has_real_feature_and_full_detection_output():
    c=safe_core();r=c.step([obs(geometry=False)],0)
    assert len(r['tracks_2d_by_camera']['1'])==1 and c.bank.objects[1].identity_embedding is not None
    r=c.step([obs(geometry=False,timestamp=1)],1);assert gid(r,1)==1


def test_long_disappearance_deletes_without_outputting_predictions():
    c=safe_core(max_missed=2);c.step([obs()],0)
    for t in range(1,4):assert c.step([],t)['tracks_2d_by_camera']=={}
    assert c.bank.objects[1].status=='DELETED'
    assert gid(c.step([obs(timestamp=4)],4),1)!=1


def test_single_camera_gap_and_two_view_update_preserve_one_object():
    c=safe_core(enable_multiview_coordination=True)
    r=c.step([obs(1),obs(2)],0);assert gid(r,1)==gid(r,2)
    for t in range(1,4):assert gid(c.step([obs(2,timestamp=t)],t),2)==1
    r=c.step([obs(1,timestamp=4),obs(2,timestamp=4)],4)
    assert gid(r,1)==gid(r,2)==1
    assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==4


def test_opposite_view_descriptors_cannot_cancel_valid_current_output():
    c=safe_core(enable_multiview_coordination=True)
    from test_worldobject_memory import evidence
    ev=evidence(0,2);ev['birth_pair_logits']=np.ones((2,2))*20
    r=c.step([obs(1),obs(2,feature=(-1,0,0))],0,evidence=ev)
    assert gid(r,1)==gid(r,2)
    assert sum(map(len,r['tracks_2d_by_camera'].values()))==2
    assert np.isfinite(c.bank.objects[1].identity_embedding).all()


def test_competing_current_people_prevent_overconfident_memory_update():
    c=safe_core();c.step([obs()],0)
    from test_worldobject_memory import evidence
    ev=evidence(1,2)
    r=c.step([obs(1,0,timestamp=1),obs(1,1,feature=(1,.01,0),timestamp=1)],1,evidence=ev)
    assert len(r['tracks_2d_by_camera']['1'])==2
    assert c.bank.objects[1].identity_state.identity_update_status=='IDENTITY_UNCERTAIN_NO_LONG_TERM_UPDATE'
    assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==0


def test_confirmed_unlike_views_survive_when_only_second_camera_remains():
    from test_worldobject_memory import evidence
    from worldmot.models.world_object_model_inputs import model_inputs
    c=safe_core(enable_multiview_coordination=True)
    ev=evidence(0,2);ev['birth_pair_logits']=np.ones((2,2))*20
    first=c.step([obs(1),obs(2,feature=(0,1,0))],0,evidence=ev)
    assert gid(first,1)==gid(first,2)==1
    memory=c.bank.objects[1].identity_state
    assert set(memory.short_term_embedding_by_camera)=={1,2}
    np.testing.assert_array_equal(memory.long_term_embedding,[1,0,0])
    current=[obs(2,feature=(0,1,0),timestamp=1)]
    tensors=model_inputs(c.bank.active(),current,c.config,1)
    np.testing.assert_array_equal(tensors[0].numpy()[0],[0,1,0])
    second=c.step(current,1)
    assert gid(second,2)==1 and c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==1
    np.testing.assert_array_equal(memory.long_term_embedding,[1,0,0])
    # Current similarity from an unrelated third appearance is insufficient,
    # even with perfectly coincident geometry.
    third=c.step([obs(2,feature=(0,0,1),timestamp=2)],2)
    assert gid(third,2)!=1


def test_unconfirmed_geometry_does_not_authorize_unlike_view_bridge():
    from test_worldobject_memory import evidence
    c=safe_core(enable_multiview_coordination=True)
    ev=evidence(0,2);ev['birth_pair_logits']=np.ones((2,2))*20
    c.step([obs(1,geometry=False),obs(2,feature=(0,1,0),geometry=False)],0,evidence=ev)
    assert set(c.bank.objects[1].identity_state.short_term_embedding_by_camera)=={1}


def test_future_camera_memory_is_not_used_by_query_or_identity_evidence():
    from worldmot.models.world_object_model_inputs import model_inputs
    from worldmot.tracking.world_object_identity_memory import identity_similarity
    c=safe_core(enable_multiview_coordination=True);c.step([obs(1)],0)
    memory=c.bank.objects[1].identity_state
    memory.short_term_embedding_by_camera[2]=np.array([0.,1,0])
    memory.short_term_confidence_by_camera[2]=1.;memory.short_term_timestamp_by_camera[2]=100
    current=obs(2,feature=(0,1,0),timestamp=1)
    np.testing.assert_array_equal(model_inputs(c.bank.active(),[current],c.config,1)[0].numpy()[0],[1,0,0])
    assert identity_similarity(c.bank.objects[1],current,c.config)==0
