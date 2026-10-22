import torch
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.tracking.persistent_query_runtime import birth_groups


def test_temporal_state_carries_gradients_and_birth_initializes_actual_observation():
    torch.manual_seed(10)
    model=PersistentWorldQueryTransformer(observation_dim=532,hidden_dim=32,max_world_queries=4,heads=4,decoder_layers=1).eval()
    first=torch.randn(1,3,532,requires_grad=True);geo=torch.randn(1,3,18)
    state=model.init_state(1);out=model(first,geo,state,camera_ids=torch.tensor([[1,2,1]]))
    xyz=torch.tensor([[[1.,2.,3.],[1.,2.,3.],[4.,5.,6.]]]);cov=torch.eye(3)[None,None].expand(1,3,3,3)
    state=model.accept_observations(out['state'],out,[{0:[0,1]}],xyz,cov)
    state.global_ids[0,0]=17
    assert torch.allclose(state.xyz[0,0],xyz[0,0])
    assert torch.allclose(state.embedding[0,0],out['observation_features'][0,:2].mean(0))
    second=model(torch.randn(1,2,532),torch.randn(1,2,18),state)
    second['assignment_logits'][0,0,0].backward()
    assert first.grad is not None and first.grad.abs().sum()>0
    # All three visual branches receive real identity-loss gradients.
    for branch in [model.rgb_projection,model.detector_projection,model.omega_projection]:
        assert sum(p.grad.abs().sum() for p in branch.parameters() if p.grad is not None)>0
    assert not state.detach().embedding.requires_grad


def test_empty_observations_do_not_create_nan_or_overwrite_world_state():
    model=PersistentWorldQueryTransformer(observation_dim=532,hidden_dim=32,max_world_queries=4,heads=4,decoder_layers=1)
    state=model.init_state(1);state.xyz[0,0]=torch.tensor([2.,3.,4.])
    out=model(torch.empty(1,0,532),torch.empty(1,0,18),state)
    assert torch.isfinite(out['assignment_logits']).all()
    new=model.accept_observations(out['state'],out,[{}],torch.empty(1,0,3))
    assert torch.equal(new.xyz,state.xyz)


def test_birth_grouping_joins_views_but_never_same_camera_detections():
    groups=birth_groups([0,1,2],torch.full((3,3),10.),[1,2,1],.7)
    assert len(groups)==2
    assert all(len({[1,2,1][i] for i in g})==len(g) for g in groups)
