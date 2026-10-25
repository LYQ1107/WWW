"""Causal TRAIN state replay using exactly the inference bank and input builder."""
from __future__ import annotations
from .world_object_identity_targets import build_identity_targets, update_history_labels
from .world_object_assignment_loss import world_object_assignment_loss
from ..models.world_object_model_inputs import model_inputs


class CausalWorldObjectTraining:
    def __init__(self, core, history_mode="predicted"):
        if history_mode not in ("predicted","teacher_forced"):
            raise ValueError("INVALID_HISTORY_MODE")
        self.core,self.history_mode=core,history_mode
        self.core.training_context=True
        self.history_labels={}

    def reset(self, scene):
        self.core.reset(scene)
        self.history_labels={}

    def step(self, observations, timestamp, labels, optimizer=None):
        if len(observations)!=len(labels):
            raise ValueError("TARGET_OBSERVATION_COUNT_MISMATCH")
        paired=sorted(zip(observations,labels),key=lambda pair:pair[0].key)
        observations=[x[0] for x in paired]
        labels=[x[1] for x in paired]
        history=self.core.bank.active()
        # First compute the model's prediction exclusively from causal state.
        outputs=self.core.forward_evidence(observations,timestamp,training=optimizer is not None)
        targets=build_identity_targets(history,observations,labels,self.history_labels,
                                       retention_steps=self.core.config.max_missed)
        inputs=model_inputs(history,observations,self.core.config,timestamp,self.core.device)
        losses=world_object_assignment_loss(outputs,targets,inputs[1])
        import torch
        if not torch.isfinite(losses["total"]):
            raise ValueError("NONFINITE_IDENTITY_TRAINING_LOSS")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
        decision=None
        if self.history_mode=="teacher_forced":
            # Teacher correction is TRAIN-only and occurs AFTER this forward.
            # Current targets never rewrite the pre-forward historical state.
            from ..tracking.joint_world_object_assignment import JointAssignmentResult,AssignmentOption
            options=[];used=set()
            for i,q in enumerate(history):
                indices=tuple(j for j,target in enumerate(targets["assignment"]) if target==i)
                if indices:
                    options.append(AssignmentOption(q.global_id,indices,0.,"TRAIN_TEACHER_HISTORY_POST_FORWARD"))
                    used.update(indices)
            births={}
            for j,label in enumerate(labels):
                if j not in used and label is not None:
                    births.setdefault(label,[]).append(j)
            for indices in births.values():
                if len({observations[j].camera_id for j in indices})==len(indices):
                    options.append(AssignmentOption(None,tuple(indices),0.,"TRAIN_TEACHER_BIRTH_POST_FORWARD"))
                    used.update(indices)
            decision=JointAssignmentResult(options,[j for j in range(len(observations)) if j not in used],
                                            {"solver":"TRAIN_TEACHER_POST_FORWARD","constraints_satisfied":True})
        result=self.core.step(observations,timestamp,evidence=outputs,training_decision=decision)
        update_history_labels(self.history_labels,self.core.bank.last_assignments,observations,labels)
        return result,{k:float(v.detach().cpu()) for k,v in losses.items()}
