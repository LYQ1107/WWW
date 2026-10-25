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
        self.last_diagnostics={}
        self.last_losses={}
        self._previous_diagnostic_ids={}
        self.last_gradient_diagnostics={}

    def reset(self, scene):
        self.core.reset(scene)
        self.history_labels={}
        self.last_diagnostics={}
        self._previous_diagnostic_ids={}

    def clip_boundary(self):
        """NumPy state is already detached; keep IDs across batching boundaries."""
        return {"bank_state":"PRESERVED","timestamp":self.core.bank.last_timestamp,
                "history_objects":len(self.core.bank.active()),"next_id":self.core.bank.next_id}

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
            norm=torch.nn.utils.clip_grad_norm_(self.core.model.parameters(),5.)
            if not torch.isfinite(norm):raise ValueError("NONFINITE_IDENTITY_GRADIENT")
            values=torch.stack([self.core.model.assignment.weight.grad.norm() if self.core.model.assignment.weight.grad is not None else norm*0,
                self.core.model.birth_pair_encoder[-1].weight.grad.norm() if self.core.model.birth_pair_encoder[-1].weight.grad is not None else norm*0,
                norm.detach()]).detach().cpu().tolist()
            self.last_gradient_diagnostics=dict(zip(("assignment_head","joint_birth_head","total_norm"),values))
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
        with torch.no_grad():
            pair_mask=torch.as_tensor(targets["joint_birth_mask"],device=self.core.device,dtype=torch.bool)
            truth=torch.as_tensor(targets["joint_birth_target"],device=self.core.device)>.5
            candidates=outputs["birth_pair_mask"]|outputs["birth_pair_mask"].T
            predicted=torch.sigmoid(outputs["birth_pair_logits"])>=self.core.config.birth_pair_threshold
            counts=torch.stack([(pair_mask&truth&candidates).sum(),(pair_mask&truth).sum(),
                (pair_mask&predicted&truth).sum(),(pair_mask&predicted&~truth).sum(),
                (pair_mask&~predicted&truth).sum(),(torch.sigmoid(outputs["birth_logits"])>=.5).sum()]).cpu().tolist()
        self.last_diagnostics={**targets["history_label_consistency_audit"],
            "birth_candidate_positive_retained":counts[0],"birth_candidate_positive_total":counts[1],
            "birth_pair_tp":counts[2],"birth_pair_fp":counts[3],"birth_pair_fn":counts[4],
            "birth_positive_predictions":counts[5],"history_objects_after":len(self.core.bank.active())}
        proposals=self.core.bank.assigner.last_birth_candidate_edges
        self.last_diagnostics["birth_proposal_positive_retained"]=sum(int(targets["joint_birth_target"][a,b]>0 or targets["joint_birth_target"][b,a]>0) for a,b in proposals)
        # Post-decision diagnostics never enter the prediction state.
        from ..evaluation.world_object_diagnostics import identity_diagnostics
        if not hasattr(self,"_previous_diagnostic_ids"):self._previous_diagnostic_ids={}
        diagnostic,_=identity_diagnostics(observations,labels,self.core.bank.last_assignments,self._previous_diagnostic_ids)
        self.last_diagnostics.update(diagnostic)
        first_split=0
        for a,b in targets["same_pairs"]:
            if targets["joint_birth_target"][a,b]>0 and self.core.bank.last_assignments[observations[a].key]!=self.core.bank.last_assignments[observations[b].key]:first_split+=1
        self.last_diagnostics["first_appearance_split_count"]=first_split
        update_history_labels(self.history_labels,self.core.bank.last_assignments,observations,labels)
        self.last_losses=losses
        keys=list(losses);values=torch.stack([losses[k].detach() for k in keys]).cpu().tolist()
        return result,dict(zip(keys,values))
