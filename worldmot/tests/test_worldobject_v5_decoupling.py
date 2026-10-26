import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from worldmot.models.world_object_observation import WorldObjectObservation
from worldmot.models.world_object_support import WorldObjectSupport
from worldmot.models.world_object_v5_geometry import WorldGeometryContext,v5_observations,projection_coverage
from worldmot.models.world_object_membership_head import WorldObjectMembershipHead,membership_inputs
from worldmot.tracking.world_object_v5_config import WorldObjectV5Config
from worldmot.tracking.world_object_birth_manager import WorldObjectBirthManager
from worldmot.tracking.world_object_historical_assignment import HistoricalWorldObjectAssignment
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.training.world_object_membership_targets import CausalMembershipLabels
from worldmot.training.world_object_membership_loss import membership_loss


def feature(index=0):
    f=np.zeros(1024,np.float32);f[index]=1;return f


def observation(t=1,camera=1,detection=0,identity=0,center=None,box=None,score=.95):
    p=np.asarray([0.,0.,5.] if center is None else center)
    points=p+np.random.default_rng(identity+123).normal(0,.02,(16,3))
    support=WorldObjectSupport(points,points,p,np.eye(3)*.01,.9,.8,"DEPTH_CLUSTERED",coordinate_frame="PERSISTENT_WORLD",depth_source="VERIFIED")
    return WorldObjectObservation(camera,detection,[40,40,40,60] if box is None else box,score,feature(identity),True,
        p,np.eye(3)*.01,support,True,.9,t,source={"alignment_confidence":1.,"alignment_residual_p95":0.})


def core(variant="B4",**kwargs):
    torch.manual_seed(100)
    cfg=WorldObjectV5Config(variant=variant,**kwargs)
    model=WorldObjectMembershipHead().eval()
    c=ObservationAnchoredWorldObjectCore(cfg,model);c.reset(1);return c


def test_membership_contract_equal_capacity_and_real_gradient():
    counts=[]
    for variant in ["B2","B3","B4"]:
        c=core(variant);counts.append(sum(p.numel() for p in c.model.parameters()))
        obs=[observation(camera=v) for v in [1,2]];c.step(obs,1,WorldGeometryContext(1))
        obs=[observation(t=2,camera=v) for v in [1,2]]
        outputs,geometry=c.forward_evidence(obs,2,WorldGeometryContext(2),training=True)
        assert set(outputs)=={"membership_logits","no_observation_logits","projected_observations","projected_history"}
        sidecar=CausalMembershipLabels();sidecar.identities[1].add("person");sidecar.anchors[1]="person"
        targets=sidecar.targets(c.bank.active(),obs,["person","person"])
        loss=membership_loss(outputs,targets);loss["total"].backward()
        assert c.model.membership.weight.grad is not None
        assert torch.isfinite(c.model.membership.weight.grad).all()
        assert c.model.membership.weight.grad.norm()>0
    assert len(set(counts))==1 and counts[0]==606468


def test_no_observation_is_per_camera_not_global_unique_detection():
    c=core();first=[observation(camera=v) for v in [1,2]];c.step(first,1,WorldGeometryContext(1))
    current=[observation(t=2,camera=v) for v in [1,2]]
    evidence={"membership_logits":torch.tensor([[5.,5.]]),"no_observation_logits":torch.tensor([[0.,0.]])}
    row=c.step(current,2,WorldGeometryContext(2),evidence=evidence)
    assert [x["global_id"] for x in row["decisions"]]==[1,1]
    assert row["bank"]["birth_candidates_in_optimizer"]==0


def test_solver_budget_fallback_preserves_historical_ids():
    c=core(solver_expansion_budget=0)
    c.step([observation()],1,WorldGeometryContext(1))
    result=c.step([observation(t=2)],2,WorldGeometryContext(2))
    assert result["bank"]["solver"]=="CONSERVATIVE_HISTORICAL_FALLBACK"
    assert result["decisions"][0]["global_id"]==1
    assert result["bank"]["births"]==0


def test_crowded_overlap_alone_does_not_suppress_birth():
    c=core();obs=[observation(identity=0),observation(identity=1,detection=1)]
    row=c.step(obs,1,WorldGeometryContext(1))
    assert len(row["tracks_2d_by_camera"]["1"])==2
    assert len(set(x["global_id"] for x in row["decisions"]))==2


def test_true_exact_duplicate_has_reason_and_no_duplicate_official_key():
    c=core();obs=[observation(),observation(detection=1)]
    row=c.step(obs,1,WorldGeometryContext(1))
    assert len(row["decisions"])==2 and len(row["tracks_2d_by_camera"]["1"])==1
    assert row["bank"]["duplicate_suppressed"]==1
    assert row["decisions"][1]["stage"]=="TRACK_AWARE_DUPLICATE"


def test_missing_world_from_omega_does_not_make_identity_projection():
    camera={"intrinsic_model":np.diag([100.,100.,1.]).tolist(),"world_from_camera":np.eye(4).tolist(),
            "original_size":[100,100],"model_size":[100,100]}
    context=WorldGeometryContext.from_record(2,{"alignment":{"success":True,"reason":"OK","scale":1.,"inlier_ratio":1.},"camera_models":{"1":camera}})
    assert context.cameras=={}
    assert projection_coverage([[0,0,5]],observation(t=2),context) is None


def test_verified_sim3_projects_persistent_support_into_current_image():
    t=np.eye(4);t[:3,:3]=np.array([[0,-2,0],[2,0,0],[0,0,2]]);t[:3,3]=[4,1,3]
    camera={"intrinsic_model":[[100,0,50],[0,100,50],[0,0,1]],"world_from_camera":np.eye(4).tolist(),
            "original_size":[100,100],"model_size":[100,100]}
    context=WorldGeometryContext.from_record(2,{"alignment":{"success":True,"reason":"OK","inlier_ratio":1.,"world_from_omega":t.tolist()},"camera_models":{"1":camera}})
    point=(t@np.array([0,0,5,1]))[:3]
    value=projection_coverage([point],observation(t=2),context)
    assert value["coverage"]==1 and context.reprojection_status=="VERIFIED_CURRENT_WORLD_CAMERA"


def test_bad_geometry_does_not_remove_correct_historical_candidate():
    c=core("B3");c.step([observation()],1,WorldGeometryContext(1,alignment_success=True,alignment_confidence=1.))
    old=c.bank.objects[1].spatial_state.state.copy()
    current=observation(t=2,center=[50,50,50]);evidence={"membership_logits":torch.tensor([[8.]]),"no_observation_logits":torch.zeros((1,2))}
    result=c.step([current],2,WorldGeometryContext(2,alignment_success=True,alignment_confidence=1.),evidence=evidence)
    assert result["decisions"][0]["global_id"]==1
    assert result["bank"]["births"]==0
    assert c.bank.objects[1].spatial_state_status.startswith("IDENTITY_RETAINED_SPATIAL_CONFLICT")
    assert c.bank.objects[1].last_identity_write==2


def test_two_coherent_measurements_reinitialize_spatial_state_without_new_id():
    c=core("B3");c.step([observation()],1,WorldGeometryContext(1,alignment_success=True,alignment_confidence=1.))
    for t in [2,3]:
        c.step([observation(t=t,center=[50,50,50])],t,WorldGeometryContext(t,alignment_success=True,alignment_confidence=1.),
               evidence={"membership_logits":torch.tensor([[8.]]),"no_observation_logits":torch.zeros((1,2))})
    assert c.bank.next_id==2 and np.allclose(c.bank.objects[1].center,[50,50,50])
    assert c.bank.objects[1].spatial_state_status=="CURRENT_EVIDENCE_SPATIAL_REINITIALIZATION_SAME_ID"


def test_lost_and_one_sided_visibility_and_past_output_immutability():
    c=core();first=c.step([observation(camera=v) for v in [1,2]],1,WorldGeometryContext(1));saved=copy.deepcopy(first)
    one=c.step([observation(t=2,camera=2)],2,WorldGeometryContext(2))
    assert one["tracks_2d_by_camera"]["2"][0]["global_id"]==1 and c.bank.objects[1].status=="CONFIRMED"
    none=c.step([],3,WorldGeometryContext(3));assert none["tracks_2d_by_camera"]=={} and c.bank.objects[1].status=="LOST"
    assert c.bank.objects[1].last_identity_write==2
    recovered=c.step([observation(t=4)],4,WorldGeometryContext(4));assert recovered["decisions"][0]["global_id"]==1
    assert first==saved


def test_wrong_reuse_keeps_original_supervision_anchor_without_repairing_bank():
    c=core();obs=[observation()];c.step(obs,1,WorldGeometryContext(1));labels=CausalMembershipLabels()
    labels.observe_after_decision(c.bank.last_assignments,obs,["A"])
    labels.observe_after_decision(c.bank.last_assignments,obs,["B"])
    labels.observe_after_decision(c.bank.last_assignments,obs,["A"])
    t=labels.targets(c.bank.active(),obs,["A"])
    assert t["history_valid"].all() and labels.identities[1]=={"A","B"}
    assert t["history_labels"]==["A"] and t["positive"][0,0]
    assert not labels.targets(c.bank.active(),obs,["B"])["positive"].any()
    assert c.bank.last_assignments[obs[0].key]==1


def test_b4_is_invariant_to_adversarial_world_geometry():
    a,b=core(),core()
    for t in [1,2,3]:
        left=[observation(t=t,camera=v) for v in [1,2]]
        right=[observation(t=t,camera=v,center=[1000*v,-999,2*t]) for v in [1,2]]
        ra=a.step(left,t,WorldGeometryContext(t,alignment_success=True,alignment_confidence=1.))
        rb=b.step(right,t,WorldGeometryContext(t,alignment_success=True,alignment_confidence=.01))
        assert ra["tracks_2d_by_camera"]==rb["tracks_2d_by_camera"]
        assert all(q.spatial_state is None for q in b.bank.active())


def test_missing_observations_do_not_modify_identity_memory():
    c=core();c.step([observation()],1,WorldGeometryContext(1));old=c.bank.objects[1].identity_embedding.copy()
    for t in range(2,8):c.step([],t,WorldGeometryContext(t))
    assert np.array_equal(c.bank.objects[1].identity_embedding,old)
    assert c.bank.objects[1].status=="LOST"


def test_dedicated_losses_cannot_request_birth_or_gt_geometry_targets():
    c=core();o=[observation()];c.step(o,1,WorldGeometryContext(1));o=[observation(t=2)]
    out,_=c.forward_evidence(o,2,WorldGeometryContext(2),training=True);s=CausalMembershipLabels();s.identities[1].add("A");s.anchors[1]="A"
    target=s.targets(c.bank.active(),o,["A"]);losses=membership_loss(out,target)
    assert set(losses)=={"per_camera_membership","detection_exclusivity","hard_negative_membership","cross_view_membership","cross_view_visual_contrast","total"}
    assert not any("birth" in k or "death" in k or "write" in k for k in c.model.state_dict())


def test_gt_changes_loss_but_never_repairs_predicted_runtime_ids():
    from worldmot.training.world_object_v5_causal_training import CausalMembershipTraining
    a,b=CausalMembershipTraining(core()),CausalMembershipTraining(core())
    a.reset(1);b.reset(1)
    for runner in [a,b]:runner.forward_loss_then_step([observation()],1,WorldGeometryContext(1),["A"])
    ra,la=a.forward_loss_then_step([observation(t=2)],2,WorldGeometryContext(2),["A"])
    rb,lb=b.forward_loss_then_step([observation(t=2)],2,WorldGeometryContext(2),["B"])
    assert ra==rb
    assert float(la["total"])!=float(lb["total"])
    assert np.array_equal(a.core.bank.objects[1].identity_embedding,b.core.bank.objects[1].identity_embedding)


def test_gate_rejects_claimed12_epochs_with_only_one_full_epoch(tmp_path):
    from worldmot.runtime.world_object_v5_contract import load_v5_membership_gate,v5_source_fingerprint
    import json
    cfg=WorldObjectV5Config()
    gate={"status":"FULL_TRAIN_GATE_PASSED","variant":"B3","v5_source_sha256":v5_source_fingerprint(),
        "tracking_config":cfg.to_dict(),"final_scenes":list(range(1,25)),"final_timestamps":28754,
        "history_mode":"predicted_from_first_timestamp","GT_in_model_inputs":False,"final_epochs":12,
        "prediction_heads":["membership","per_camera_no_observation"],"final_total_training_timestamps":28754*12,
        "final_epoch_coverage":[[{"scene":s,"timestamps":1} for s in range(1,25)]]}
    path=tmp_path/"gate.json";path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match="EVERY_FINAL_EPOCH_COVERAGE_REQUIRED"):load_v5_membership_gate(path,cfg)


def test_gradient_truncation_boundary_preserves_causal_identity_bank():
    from worldmot.training.world_object_v5_causal_training import CausalMembershipTraining
    c=core();runner=CausalMembershipTraining(c);runner.reset(1)
    for t in range(1,66):
        runner.forward_loss_then_step([observation(t=t)],t,WorldGeometryContext(t),["A"])
        if t==64:
            before=(c.bank.next_id,c.bank.objects[1].hits,c.bank.last_timestamp)
            boundary=runner.clip_boundary()
            assert boundary["bank_state"]=="PRESERVED"
            assert (c.bank.next_id,c.bank.objects[1].hits,c.bank.last_timestamp)==before
    assert c.bank.next_id==2 and c.bank.objects[1].hits==65
