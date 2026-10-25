import numpy as np
from worldobject_synthetic import obs,core,gid


def evidence(m,n,dim=3):
    return {"assignment_logits":np.ones((m,n))*3,"null_logits":np.zeros(m),
        "existence_logits":np.ones(m)*5,"birth_logits":np.ones(n)*5,
        "update_gate":np.ones((m,n)),"memory_delta":np.zeros((m,n,dim))}


def test_alignment_failure_cannot_overwrite_spatial_state():
    c=core();c.step([obs()],0)
    previous=c.bank.objects[1].center.copy()
    r=c.step([obs(center=(999,999,999),geometry=False,timestamp=1)],1)
    assert gid(r,1)==1
    np.testing.assert_array_equal(previous,c.bank.objects[1].center)
    assert r["decisions"][0]["geometry_status"]=="WORLD_ALIGNMENT_FAILED"


def test_unmatched_and_low_confidence_features_do_not_rewrite_other_identity():
    c=core();c.step([obs()],0)
    previous=c.bank.objects[1].identity_embedding.copy()
    c.step([obs(1,0,feature=(0,1,0),timestamp=1)],1)
    np.testing.assert_array_equal(previous,c.bank.objects[1].identity_embedding)
    c.step([obs(1,0,feature=(1,.3,0),timestamp=2,score=.1)],2)
    np.testing.assert_array_equal(previous,c.bank.objects[1].identity_embedding)


def test_update_gate_and_memory_delta_are_used_and_read_next_step():
    c=core();c.step([obs()],0)
    old=c.bank.objects[1].identity_embedding.copy()
    ev=evidence(1,1);ev["update_gate"][:]=0;ev["memory_delta"][:]=[0,5,0]
    c.step([obs(timestamp=1)],1,evidence=ev)
    np.testing.assert_array_equal(old,c.bank.objects[1].identity_embedding)
    ev["update_gate"][:]=1
    c.step([obs(timestamp=2)],2,evidence=ev)
    assert c.bank.objects[1].identity_embedding[1]>0
    from worldmot.models.world_object_model_inputs import model_inputs
    inputs=model_inputs(c.bank.active(),[obs(timestamp=3)],c.config,3)
    np.testing.assert_allclose(inputs[0].numpy()[0],c.bank.objects[1].identity_embedding)


def test_birth_null_existence_and_death_outputs_affect_decisions():
    c=core();ev=evidence(0,1);ev["birth_logits"][:]=-20
    r=c.step([obs()],0,evidence=ev)
    assert r["tracks_2d_by_camera"]["1"][0]["status"]=="UNCONFIRMED"
    ev=evidence(1,1);ev["assignment_logits"][:]=-30;ev["null_logits"][:]=30
    r=c.step([obs(timestamp=1)],1,evidence=ev)
    assert gid(r,1)!=1
    ev=evidence(2,0);ev["existence_logits"][:]=-30
    c.step([],2,evidence=ev)
    assert c.bank.objects[1].status=="DELETED"


def test_strong_cross_view_visual_recovery_is_not_blocked_by_stale_position():
    c=core();c.step([obs()],0);c.step([],1)
    previous=c.bank.objects[1].center.copy()
    r=c.step([obs(2,center=(100,0,3),timestamp=2)],2)
    assert gid(r,2)==1
    np.testing.assert_array_equal(previous,c.bank.objects[1].center)
