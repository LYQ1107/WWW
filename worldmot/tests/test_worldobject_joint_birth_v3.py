from dataclasses import replace
import numpy as np
import pytest
import torch
from worldobject_synthetic import obs,core,gid
from test_worldobject_memory import evidence
from worldmot.models.world_object_update_head import WorldObjectUpdateHead
from worldmot.models.world_object_model_inputs import model_inputs
from worldmot.training.world_object_identity_targets import build_identity_targets
from worldmot.training.world_object_assignment_loss import world_object_assignment_loss
from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
from worldmot.tracking.world_object_config import WorldObjectConfig


def pair_evidence(n,score):
    result=evidence(0,n);result["birth_pair_logits"]=np.full((n,n),-20.)
    result["birth_pair_logits"][0,1]=result["birth_pair_logits"][1,0]=score
    return result


def test_empty_history_same_person_uses_learned_birth_not_cosine_gate():
    c=core();r=c.step([obs(1,feature=(1,0,0)),obs(2,feature=(0,1,0))],0,evidence=pair_evidence(2,10.))
    assert gid(r,1)==gid(r,2)
    assert r["decisions"][0]["decision"]=="LEARNED_CROSS_VIEW_JOINT_BIRTH"


def test_strong_joint_birth_is_not_pruned_by_low_singleton_birth_prior():
    c=core();ev=pair_evidence(2,10.);ev["birth_logits"]=np.full(2,-10.)
    r=c.step([obs(1),obs(2)],0,evidence=ev)
    assert gid(r,1)==gid(r,2)
    assert (0,1) in c.bank.assigner.last_birth_candidate_edges
    assert all(len(v)==1 for v in r["tracks_2d_by_camera"].values())


def test_soft_birth_prior_still_lets_history_outbid_new_birth():
    c=core();c.step([obs(1)],0)
    ev=evidence(1,2);ev["birth_logits"]=np.full(2,-10.)
    ev["birth_pair_logits"]=np.full((2,2),10.)
    r=c.step([obs(1,timestamp=1),obs(2,timestamp=1)],1,evidence=ev)
    assert gid(r,1)==gid(r,2)==1 and c.bank.next_id==2


def test_pair_topk_uses_pair_confidence_before_singleton_prior():
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    c=WorldObjectTrackingCore(WorldObjectConfig(3,top_k=1));c.reset(1)
    ev=evidence(0,3);ev["birth_logits"]=np.array([10.,-10.,10.])
    ev["birth_pair_logits"]=np.full((3,3),-30.)
    ev["birth_pair_logits"][0,1]=ev["birth_pair_logits"][1,0]=10.
    ev["birth_pair_logits"][0,2]=ev["birth_pair_logits"][2,0]=9.
    result=c.step([obs(1),obs(2,0),obs(2,1)],0,evidence=ev)
    assert c.bank.assigner.last_birth_candidate_edges=={(0,1)}
    assert gid(result,1)==gid(result,2,0) and gid(result,1)!=gid(result,2,1)


@pytest.mark.parametrize("method",["FULL_WORLD_OBJECT_V3","NO_LEARNED_JOINT_BIRTH_V3"])
def test_learned_current_identity_evidence_can_jointly_update_history_for_dissimilar_views(method):
    c=core(method);c.step([obs(1)],0)
    ev=evidence(1,2);ev["assignment_logits"]=np.full((1,2),6.)
    ev["null_logits"]=np.full(1,-4.);ev["birth_logits"]=np.full(2,-10.)
    ev["birth_pair_logits"]=np.full((2,2),10.)
    result=c.step([obs(1,timestamp=1),obs(2,feature=(0,1,0),timestamp=1)],1,evidence=ev)
    assert gid(result,1)==gid(result,2)==1 and c.bank.next_id==2
    assert result["decisions"][0]["decision"]=="JOINT_HISTORY_TWO_VIEW"


@pytest.mark.parametrize("m,n,budget,seed",[(0,0,4096,0),(1,2,4096,0),(2,4,4096,1),
    (7,20,4096,2),(20,64,4096,3),(60,100,4096,4),(90,120,4096,5),
    (50,70,8,6),(15,40,1,7),(10,20,100000,8)])
def test_compact_beam_matches_frozen_frozenset_reference(m,n,budget,seed):
    import math
    from worldmot.tracking.joint_world_object_assignment import JointWorldObjectAssignment,AssignmentOption
    rng=np.random.default_rng(seed);config=replace(WorldObjectConfig(3),max_exact_states=budget)
    groups=[]
    for i in range(m):
        indices=rng.choice(n,min(4,n),replace=False).tolist()
        options=[AssignmentOption(i+1,(j,),float(rng.integers(1,4)),"SINGLE") for j in indices]
        options += [AssignmentOption(i+1,tuple(sorted((a,b))),float(rng.integers(2,5)),"PAIR") for a in indices for b in indices if a<b]
        groups.append(options)
    exact=math.prod(len(g)+1 for g in groups)<=budget;width=budget if exact else config.beam_width
    states=[(0.,frozenset(),())];expanded=0
    for group in groups:
        next_states=[]
        for value,used,chosen in states:
            next_states.append((value,used,chosen))
            for option in group:
                expanded+=1
                if not used.intersection(option.indices):
                    next_states.append((value+option.benefit,used.union(option.indices),chosen+(option,)))
        next_states.sort(key=lambda x:(-x[0],tuple((o.global_id or 0,o.indices) for o in x[2])))
        states=next_states[:width]
    actual,audit=JointWorldObjectAssignment(config)._solve_component(groups)
    assert actual==list(states[0][2]) and audit["expanded_states"]==expanded
    assert audit["mode"]==("EXACT" if exact else "BEAM")


def test_positive_birth_proposal_diagnostics_are_json_serializable():
    import json
    c=core();c.model=WorldObjectUpdateHead(3,8)
    with torch.no_grad():
        c.model.birth_pair_encoder[-1].bias.fill_(10)
        c.model.lifecycle[-1].bias[0]=10
    runner=CausalWorldObjectTraining(c)
    runner.step([obs(1),obs(2)],0,["A","A"])
    assert runner.last_diagnostics["birth_proposal_positive_retained"]==1
    json.dumps(runner.last_diagnostics,allow_nan=False)


def test_low_joint_birth_logit_keeps_two_boxes_separate():
    c=core();r=c.step([obs(1),obs(2)],0,evidence=pair_evidence(2,-10.))
    assert gid(r,1)!=gid(r,2) and sum(map(len,r["tracks_2d_by_camera"].values()))==2


def test_competing_birth_pairs_choose_one_detection_once():
    c=core();observations=[obs(1),obs(2,0,feature=(0,1,0)),obs(2,1)]
    ev=evidence(0,3);ev["birth_pair_logits"]=np.full((3,3),-20.)
    ev["birth_pair_logits"][0,1]=1.;ev["birth_pair_logits"][0,2]=12.
    r=c.step(observations,0,evidence=ev)
    assert gid(r,1)==gid(r,2,1) and gid(r,1)!=gid(r,2,0)
    assert len(c.bank.last_assignments)==3


def test_reliable_spatial_conflict_can_veto_birth_even_with_strong_visual_score():
    c=core();r=c.step([obs(1),obs(2,center=(20,0,3))],0,evidence=pair_evidence(2,20.))
    assert gid(r,1)!=gid(r,2)


def test_missing_or_weak_geometry_allows_learned_visual_birth_without_writing_space():
    c=core();r=c.step([obs(1,geometry=False),obs(2,geometry=False)],0,evidence=pair_evidence(2,10.))
    assert gid(r,1)==gid(r,2) and not c.bank.objects[gid(r,1)].geometry_valid


def test_weak_depth_support_is_available_but_not_strong_fusion_or_update():
    o=obs();support=replace(o.support,ambiguity="DEPTH_NOT_SEPARATED",quality=.9)
    weak=replace(o,spatial_support=support,cross_view_fusion_eligible=None,world_state_update_eligible=None)
    assert weak.point_support_available and weak.geometry_valid
    assert not weak.cross_view_fusion_eligible and not weak.world_state_update_eligible
    c=core();c.step([obs()],0);before=c.bank.objects[1].center.copy()
    weak=replace(weak,world_center=np.array([1000.,0,3]),timestamp=1)
    r=c.step([weak],1)
    assert gid(r,1)==1
    np.testing.assert_array_equal(before,c.bank.objects[1].center)


def test_joint_birth_targets_ignore_unmatched_same_camera_and_existing_identity():
    observations=[obs(1),obs(2,0),obs(2,1)]
    targets=build_identity_targets([],observations,["A",None,"A"],{})
    assert targets["joint_birth_target"][0,2]==1 and not targets["joint_birth_mask"][0,1]
    c=core();c.step([obs()],0)
    old=build_identity_targets(c.bank.active(),observations,["A","B","A"],{1:"A"})
    assert not old["joint_birth_mask"][0,2] and old["joint_birth_mask"][0,1]


@pytest.mark.parametrize("labels",[["A","B","A"],["A","B","C"]])
def test_birth_pair_head_has_finite_nonzero_gradient_including_no_positive_batch(labels):
    torch.manual_seed(9)
    c=core();observations=[obs(1),obs(2,0,feature=(0,1,0)),obs(2,1)]
    model=WorldObjectUpdateHead(3,16)
    inputs=model_inputs([],observations,c.config,0);outputs=model(*inputs)
    targets=build_identity_targets([],observations,labels,{})
    loss=world_object_assignment_loss(outputs,targets,inputs[1])
    loss["total"].backward()
    gradient=model.birth_pair_encoder[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert outputs["birth_pair_logits"].shape==(3,3)


def test_no_learned_birth_reuses_full_capacity_but_disables_decision_evidence():
    c=core("NO_LEARNED_JOINT_BIRTH_V3")
    r=c.step([obs(1,feature=(1,0,0)),obs(2,feature=(0,1,0))],0,evidence=pair_evidence(2,20.))
    assert gid(r,1)!=gid(r,2)
    assert c.config.checkpoint_method_id=="FULL_WORLD_OBJECT_V3"


def test_capacity_alias_canonicalizes_to_one_experiment():
    a=WorldObjectConfig(3,"APPEARANCE_CAPACITY_MATCHED_V3")
    b=WorldObjectConfig(3,"FULL_NO_GEOMETRY_V3")
    assert a==b


def test_continuous_history_survives_64_and_128_boundaries_and_scene_reset():
    torch.manual_seed(8);c=core();c.model=WorldObjectUpdateHead(3,8)
    with torch.no_grad():
        c.model.assignment.bias.fill_(10)
        c.model.lifecycle[-1].bias[0]=-10
    runner=CausalWorldObjectTraining(c);runner.reset(1)
    ids=[]
    for t in range(260):
        if t and t%64==0:
            snapshot=runner.clip_boundary();assert snapshot["next_id"]==2
        r,_=runner.step([obs(timestamp=t)],t,["A"]);ids.append(gid(r,1))
    assert set(ids)=={1} and c.bank.objects[1].age==260
    runner.reset(2);assert not c.bank.objects and not runner.history_labels


def test_duplicate_predicted_history_identity_is_multi_positive_not_false_birth():
    c=core();c.step([obs(1,0),obs(1,1)],0)
    targets=build_identity_targets(c.bank.active(),[obs(timestamp=1)],["A"],{1:"A",2:"A"})
    assert targets["assignment"][0]==-2 and targets["observation_valid"][0]
    assert targets["assignment_positive"][:,0].tolist()==[True,True,False]
    assert targets["birth"][0]==0


@pytest.mark.parametrize("m,n",[(0,0),(0,2),(2,0)])
def test_empty_shape_combinations_have_finite_loss(m,n):
    c=core()
    if m:c.step([obs(1,i) for i in range(m)],0)
    observations=[obs(1 if j==0 else 2,j,timestamp=1) for j in range(n)]
    model=WorldObjectUpdateHead(3,8);inputs=model_inputs(c.bank.active(),observations,c.config,1)
    outputs=model(*inputs)
    targets=build_identity_targets(c.bank.active(),observations,[str(i) for i in range(n)],{i+1:str(i) for i in range(m)})
    losses=world_object_assignment_loss(outputs,targets,inputs[1])
    assert torch.isfinite(losses["total"])


def test_birth_candidate_edges_are_bounded_without_ground_truth_pruning():
    c=core(max_birth_pair_edges=1)
    with pytest.raises(ValueError,match="BIRTH_PAIR_EDGE_LIMIT"):
        model_inputs([],[obs(1),obs(2,0),obs(2,1)],c.config,0)


def test_vectorized_spatial_distance_matches_scalar_uncertainty_rule():
    from worldmot.tracking.joint_world_object_assignment import spatial_distance,spatial_distance_matrix
    a=[obs(1,center=(i*.2,0,3)) for i in range(3)]
    b=[obs(2,center=(i*.1,.2,3)) for i in range(4)]
    matrix=spatial_distance_matrix(a,b)
    expected=np.array([[spatial_distance(x.world_center,x.world_covariance,y.world_center,y.world_covariance) for y in b] for x in a])
    np.testing.assert_allclose(matrix,expected,rtol=1e-7)


def test_low_quality_nonfinite_geometry_cannot_break_valid_candidate_graph():
    c=core();bad=obs(2,center=(np.nan,0,3),geometry=False)
    result=c.step([obs(1),bad],0)
    assert sum(map(len,result["tracks_2d_by_camera"].values()))==2


def test_single_birth_is_conditioned_on_real_history_not_current_roi_alone():
    c=core();model=WorldObjectUpdateHead(3,8);seen=[]
    handle=model.birth.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    model(*model_inputs([],[obs()],c.config,0))
    c.step([obs()],0)
    model(*model_inputs(c.bank.active(),[obs(timestamp=1)],c.config,1))
    handle.remove()
    assert seen[0].shape[1]==16 and seen[0][0,-4]==0 and seen[1][0,-4]==1


def test_strong_learned_history_evidence_can_recover_dissimilar_camera_roi():
    c=core();c.step([obs(1)],0)
    ev=evidence(1,1);ev["assignment_logits"][:]=10.;ev["birth_logits"][:]=-10.
    result=c.step([obs(2,feature=(0,1,0),timestamp=1)],1,evidence=ev)
    assert gid(result,2)==1
