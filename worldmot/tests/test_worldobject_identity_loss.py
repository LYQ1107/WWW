import numpy as np
import torch
from worldobject_synthetic import obs,core
from worldmot.models.world_object_update_head import WorldObjectUpdateHead
from worldmot.models.world_object_model_inputs import model_inputs
from worldmot.training.world_object_identity_targets import build_identity_targets,match_train_identities
from worldmot.training.world_object_assignment_loss import world_object_assignment_loss
from worldmot.training.world_object_causal_training import CausalWorldObjectTraining


def test_assignment_loss_has_history_specific_gradient_and_memory_gradient():
    torch.manual_seed(7)
    c=core();c.step([obs(1,0),obs(1,1,feature=(0,1,0))],0)
    observations=[obs(1,0,timestamp=1),obs(1,1,feature=(0,1,0),timestamp=1),obs(2,0,timestamp=1)]
    inputs=model_inputs(c.bank.active(),observations,c.config,1)
    model=WorldObjectUpdateHead(3,16)
    outputs=model(*inputs);outputs["assignment_logits"].retain_grad()
    targets=build_identity_targets(c.bank.active(),observations,["A","B","A"],{1:"A",2:"B"})
    losses=world_object_assignment_loss(outputs,targets,inputs[1]);losses["total"].backward()
    grad=outputs["assignment_logits"].grad
    assert torch.isfinite(grad).all() and not torch.allclose(grad[0],grad[1])
    assert model.assignment.weight.grad.abs().sum()>0
    assert model.memory.weight.grad.abs().sum()>0
    assert model.birth[-1].weight.grad.abs().sum()>0
    assert model.lifecycle[-1].weight.grad.abs().sum()>0
    assert model.update.weight.grad.abs().sum()>0


def test_predicted_training_uses_same_core_and_resets_clip_history():
    torch.manual_seed(3)
    c=core();c.model=WorldObjectUpdateHead(3,16)
    trainer=CausalWorldObjectTraining(c,"predicted");trainer.reset(1)
    _,losses=trainer.step([obs()],0,["A"])
    assert np.isfinite(losses["total"])
    trainer.step([obs(timestamp=1)],1,["A"])
    trainer.reset(2)
    assert trainer.history_labels=={} and trainer.core.bank.objects=={}


def test_duplicate_train_identity_and_annotation_id_are_not_supervised_as_identity():
    o=[obs()]
    anns={1:[{"id":91,"instance_id":2,"bbox":[10,10,10,20]},
             {"id":92,"instance_id":2,"bbox":[10,10,10,20]}]}
    labels,audit=match_train_identities(o,anns)
    assert labels==[None] and audit["ambiguous_or_invalid_annotations"]==2
    labels,_=match_train_identities(o,{1:[{"id":91,"bbox":[10,10,10,20]}]})
    assert labels==[None]


def test_teacher_correction_happens_only_after_forward():
    c=core();model=WorldObjectUpdateHead(3,16);c.model=model
    trainer=CausalWorldObjectTraining(c,"teacher_forced");trainer.reset(1)
    observed=[]
    handle=model.register_forward_pre_hook(lambda module,inputs:observed.append(inputs[0].detach().clone()))
    trainer.step([obs()],0,["A"])
    assert observed[0].shape==(0,3)
    assert not hasattr(c.bank.objects[1],"gt_identity")
    handle.remove()
