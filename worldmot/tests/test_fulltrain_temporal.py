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


def test_checkpoint_resume_preserves_random_streams_and_optimizer(tmp_path):
    import random
    import numpy as np
    from worldmot.training.train_utils import seed_everything, save_checkpoint, restore_rng_state
    seed_everything(20261022)
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters())
    model(torch.randn(4,3)).sum().backward();opt.step()
    path=tmp_path/'resume.pt';save_checkpoint(path,model,opt,1,{'scope':'regression'})
    expected=(random.random(),np.random.random(),torch.rand(5))
    seed_everything(7)
    payload=torch.load(path,weights_only=True)
    restored=torch.nn.Linear(3,2);restored.load_state_dict(payload['model'])
    restored_opt=torch.optim.AdamW(restored.parameters());restored_opt.load_state_dict(payload['optimizer'])
    restore_rng_state(payload['rng_state'])
    assert random.random()==expected[0]
    assert np.random.random()==expected[1]
    assert torch.equal(torch.rand(5),expected[2])
    assert len(restored_opt.state)==len(opt.state)>0


def test_geometry_ablation_retains_omega_visual_features_without_geometry_leakage():
    torch.manual_seed(13)
    model=PersistentWorldQueryTransformer(observation_dim=532,hidden_dim=32,max_world_queries=4,
        heads=4,decoder_layers=1,ablation='no_geometry').eval()
    observations=torch.randn(1,3,532);geometry=torch.randn(1,3,18)
    baseline=model(observations,geometry)
    changed=observations.clone();changed[...,512:530]=torch.randn(1,3,18)*100;changed[...,-1]=100
    without_geometry=model(changed,geometry*100+17)
    for key in ['assignment_logits','pair_logits','birth_logits']:
        assert torch.allclose(baseline[key],without_geometry[key])
    changed[...,:512]=observations[...,:512];changed[...,256:512]=torch.randn(1,3,256)
    with_different_omega=model(changed,geometry)
    assert not torch.allclose(baseline['observation_features'],with_different_omega['observation_features'])


def test_similarity_scale_does_not_scale_camera_rotation(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from worldmot.models import world_observation_encoder as module
    from worldmot.models.person_detector import PersonDetection
    from worldmot.geometry.camera_model import CameraModel
    from worldmot.geometry.world_alignment import Sim3
    feature=SimpleNamespace(appearance_feature=np.ones(512,np.float32),omega_feature=np.ones(256),
        detector_feature=np.ones(128),valid=True,source='regression')
    extractor=SimpleNamespace(extract=lambda *args:{(1,0):feature})
    lifted=SimpleNamespace(center=np.array([0.,0.,5.]),covariance=np.eye(3),
        geometry_reliable=True,rejection_reason='',support_points=np.array([[0.,0.,5.]]))
    monkeypatch.setattr(module,'lift_detection',lambda *args:lifted)
    camera=CameraModel(np.diag([100.,100.,1.]),np.eye(4),(8,8),(8,8))
    frame=SimpleNamespace(cameras={1:camera})
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    transform=Sim3(3.,rotation,np.array([2.,3.,4.]))
    encoder=module.MultiViewObservationEncoder(feature_extractor=extractor)
    token=encoder.encode({1:[PersonDetection(1,0,(0,0,4,4),.9)]},
        {1:np.zeros((8,8,3),np.uint8)},frame,1,transform)[0]
    assert np.allclose(token.geometry_feature[12:15],[0,0,1])
    assert np.allclose(token.covariance,9*np.eye(3))
    assert np.array_equal(camera.world_from_camera,np.eye(4))
