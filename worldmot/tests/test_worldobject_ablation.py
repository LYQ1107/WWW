import numpy as np
import pytest
from worldobject_synthetic import obs,core,gid
from worldmot.models.world_object_model_inputs import model_inputs


@pytest.mark.parametrize("method",["FULL_NO_GEOMETRY_V3","APPEARANCE_CAPACITY_MATCHED_V3"])
def test_entire_no_geometry_path_is_invariant_to_arbitrary_geometry(method):
    a,b=core(method),core(method)
    first=a.step([obs(1),obs(2)],0)
    second=b.step([obs(1,center=(999,99,-5),geometry=False),obs(2,center=(-999,5,9),geometry=False)],0)
    assert first["tracks_2d_by_camera"]==second["tracks_2d_by_camera"]
    assert all(not q.geometry_valid for q in a.bank.active()+b.bank.active())
    i=model_inputs(a.bank.active(),[obs(timestamp=1)],a.config,1)
    j=model_inputs(b.bank.active(),[obs(timestamp=1,center=(1e8,0,0),geometry=False)],b.config,1)
    for x,y in zip(i,j):
        np.testing.assert_array_equal(x.numpy(),y.numpy())


def test_head_only_ablation_retains_upstream_spatial_decisions():
    c=core("HEAD_ONLY_NO_GEOMETRY_V3")
    r=c.step([obs(1),obs(2,center=(20,0,3))],0)
    assert gid(r,1)!=gid(r,2)
    inputs=model_inputs(c.bank.active(),[obs(timestamp=1)],c.config,1)
    assert not inputs[2].numpy().any()
    assert c.bank.objects[1].geometry_valid


def test_capacity_matched_methods_have_identical_parameter_counts():
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    counts=[sum(p.numel() for p in WorldObjectUpdateHead(3,16).parameters()) for _ in range(4)]
    assert len(set(counts))==1
