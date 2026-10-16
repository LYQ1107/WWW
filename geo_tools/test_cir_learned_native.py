"""Learned native interface tests with controlled real risk modules/Instances.

Manual probe weights exercise the interface; they are not research models.
"""
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import tempfile
from pathlib import Path
import unittest
import torch
import common
from test_cir_guard_native import fixture,run
from gtr.modeling.identity_revision.controller import IdentityRevisionController
from gtr.modeling.identity_revision.model import NativeRiskHead,INPUT_CONTRACT
from gtr.modeling.identity_revision.state_transaction import fingerprint


def risk_fixture():
    m,o,s=fixture();c=m.cir_controller;c.mode='risk_mlp'
    with torch.random.fork_rng(devices=[]):head=NativeRiskHead('ordinary_mlp')
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head.network[0].weight[0,2]=1;head.network[2].weight[0,0]=1
        head.network[4].weight[0,0]=-1;head.network[4].bias[0]=.5
    c.risk_head=head.eval().requires_grad_(False);c.risk_threshold=0.
    return m,o,s


class LearnedNativeContracts(unittest.TestCase):
    def test_real_risk_head_veto_joint_solver_and_gallery(self):
        m,o,s=risk_fixture();seen=[]
        hook=m.cir_controller.risk_head.register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
        try:result=run(m,o,s)
        finally:hook.remove()
        self.assertEqual(result[0][-1].track_ids.tolist(),[1]);self.assertEqual(o.counts,{1:4,2:3})
        self.assertEqual(m.cir_controller.veto_edges,1)
        self.assertEqual(m.cir_controller.risk_forward_calls,2)
        self.assertEqual(m.cir_controller.guard_changed_accepted_rows,1)
        self.assertGreaterEqual(len(seen),2)
        # The rejected competitor is a solver constraint, never a million-size
        # fake risk margin in subsequent model evidence.
        self.assertTrue(all(float(x[:,1].abs().max())<1 for x in seen))
    def test_no_accepted_native_edge_cannot_be_vetoed(self):
        m,o,s=risk_fixture();result=run(m,o,torch.zeros_like(s))
        self.assertEqual(result[0][-1].track_ids.tolist(),[3])
        self.assertEqual(m.cir_controller.risk_forward_calls,0);self.assertEqual(m.cir_controller.veto_edges,0)
    def test_unchanged_proposal_is_not_validated_twice(self):
        m,o,s=risk_fixture();m.cir_controller.risk_threshold=10.
        self.assertEqual(run(m,o,s)[0][-1].track_ids.tolist(),[2])
        self.assertEqual(m.cir_controller.risk_forward_calls,1)
        self.assertEqual(m.cir_controller.guard_calls_with_veto,0)
    def test_reference_rule_same_legal_solver_path(self):
        m,o,s=fixture();m.cir_controller.mode='guard_reference'
        result=run(m,o,s);self.assertEqual(result[0][-1].track_ids.tolist(),[1])
        self.assertEqual(m.cir_controller.guard_joint_solves,1)
    def test_missing_or_masked_reference_is_not_neural_evidence(self):
        m,o,s=risk_fixture();c=m.cir_controller;matrix=torch.tensor([[.6,.61]])
        args=(torch.tensor([2]),o.instances,6,torch.tensor([1,2]),torch.tensor([.3,.3]))
        with self.assertRaises(ValueError):c.precommit(matrix,*args)
        bad=matrix.clone();bad[0,0]=-1e6
        with self.assertRaises(ValueError):c.precommit(matrix,*args,reference_scores=bad)
    def test_fault_restores_frozen_head_and_native_state(self):
        m,o,s=risk_fixture();c=m.cir_controller;before=fingerprint(c.risk_head.state_dict());rng=torch.get_rng_state().clone()
        original=m.commit_native_identity_state;calls=[]
        def commit(*args):
            result=original(*args);calls.append(1)
            if len(calls)==1:
                with torch.no_grad():c.risk_head.network[0].weight.add_(999)
                torch.rand(4);raise RuntimeError('injected head/Gallery corruption after commit')
            return result
        m.commit_native_identity_state=commit;result=run(m,o,s)
        self.assertTrue(c.failed);self.assertEqual(result[0][-1].track_ids.tolist(),[2])
        self.assertEqual(o.counts,{1:3,2:4});self.assertEqual(before,fingerprint(c.risk_head.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_missing_checkpoint_and_wrong_architecture_rejected(self):
        cfg=dict(ENABLED=True,MODE='risk_mlp',ENABLE_PROVISIONAL=False,ENABLE_REVISION=False,ENABLE_RELINK=False,REVISION_WINDOW=0)
        with self.assertRaises(ValueError):IdentityRevisionController(SimpleNamespace(**cfg))
        with tempfile.TemporaryDirectory() as tmp:
            with torch.random.fork_rng(devices=[]):head=NativeRiskHead('ordinary_mlp')
            path=Path(tmp)/'head.pt';torch.save(dict(input_contract=INPUT_CONTRACT,architecture='ordinary_mlp',state_dict=head.state_dict(),threshold=0.),path)
            cfg.update(MODE='risk_grouped',MODEL_PATH=str(path),MODEL_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):IdentityRevisionController(SimpleNamespace(**cfg))
    def test_audit_hook_survives_snapshot_and_is_removed_from_restored_clone(self):
        from cir_risk_trace import RiskTrace
        m,o,s=risk_fixture();c=m.cir_controller
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'probe.pt';torch.save(dict(input_contract=INPUT_CONTRACT,architecture='ordinary_mlp',state_dict=c.risk_head.state_dict(),threshold=0.),path)
            trace=RiskTrace(c,path,hashlib.sha256(path.read_bytes()).hexdigest());trace.install()
            original=m.commit_native_identity_state;calls=[]
            def commit(*args):
                result=original(*args);calls.append(1)
                if len(calls)==1:raise RuntimeError('injected real commit failure with audit observer active')
                return result
            m.commit_native_identity_state=commit
            try:
                run(m,o,s);self.assertTrue(c.failed);self.assertGreater(trace.stats['model_forward_calls'],0)
            finally:trace.close()
            self.assertEqual(len(c.risk_head._forward_hooks),0)


if __name__=='__main__':unittest.main()
