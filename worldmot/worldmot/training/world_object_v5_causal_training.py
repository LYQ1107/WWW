"""Predicted histories from the first timestamp; labels only build loss/audits."""
from collections import Counter
import torch
from .world_object_membership_targets import CausalMembershipLabels
from .world_object_membership_loss import membership_loss
from ..evaluation.world_object_diagnostics import identity_diagnostics


class CausalMembershipTraining:
    def __init__(self,core):
        self.core=core;self.labels=CausalMembershipLabels();self.previous={};self.last_diagnostics={}

    def reset(self,scene):
        self.core.reset(scene);self.labels=CausalMembershipLabels();self.previous={}

    def clip_boundary(self):
        return {"bank_state":"PRESERVED","next_id":self.core.bank.next_id,"timestamp":self.core.bank.last_timestamp,
                "history_objects":len(self.core.bank.active()),"GT_repair":False}

    def step(self,observations,timestamp,context,*,training=False):
        raise RuntimeError("Use forward_loss_then_step with TRAIN labels explicitly outside the core")

    def forward_loss_then_step(self,observations,timestamp,context,labels,*,training=False):
        history=self.core.bank.active()
        outputs,geometry=self.core.forward_evidence(observations,timestamp,context,training=training)
        targets=self.labels.targets(history,observations,labels)
        losses=membership_loss(outputs,targets)
        if not bool(torch.isfinite(losses["total"])):raise ValueError("NONFINITE_V5_MEMBERSHIP_LOSS")
        result=self.core.step(observations,timestamp,context,evidence=outputs,geometry=geometry)
        diagnostic,_=identity_diagnostics(observations,labels,self.core.bank.last_assignments,self.previous)
        self.last_diagnostics={**diagnostic,"mixed_historical_objects_before":targets["mixed_historical_objects"],
            "history_objects_before":len(history),"anchored_supervised_history_objects":int(targets["history_valid"].sum()),
            "pure_supervised_history_objects":int(targets["pure_history_valid"].sum()),
            "clean_supervised_observations":int(targets["observation_valid"].sum()),
            "correct_historical_membership":sum(labels[j] is not None and any(q.global_id==self.core.bank.last_assignments[o.key] and targets["positive"][i,j] for i,q in enumerate(history)) for j,o in enumerate(observations)),
            "birth_count":result["bank"]["births"],"duplicate_suppressed":result["bank"]["duplicate_suppressed"],
            "LOST_recovery_count":result["bank"].get("recoveries",0)}
        self.labels.observe_after_decision(self.core.bank.last_assignments,observations,labels)
        return result,losses
