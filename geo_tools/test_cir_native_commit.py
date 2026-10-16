"""Original source versus refactored real native methods (CPU, no fake Instances)."""
import ast
import copy
import subprocess
from types import FunctionType, MethodType, SimpleNamespace
import unittest
import torch
import common
from detectron2.structures import Boxes, Instances
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.identity_revision.controller import IdentityRevisionController
from gtr.modeling.identity_revision.state_transaction import fingerprint

BASE = 'f595189e62dac917a4880b1220b3c9ae6d7303aa'


def original_method():
    text = subprocess.check_output(['git', 'show', BASE+':gtr/modeling/meta_arch/gtr_rcnn.py'], cwd=common.ROOT,text=True)
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GTRRCNN')
    function = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_global_tracker_plus')
    unit = ast.Module(body=[function], type_ignores=[])
    context = dict(vars(native))
    exec(compile(unit, '<frozen native source>', 'exec'), context)
    # Keep original code, but use the real module namespace so a read-only
    # solver observer sees the call just as it does for the modified method.
    return FunctionType(context['run_global_tracker_plus'].__code__, vars(native))


def instance(ids=None, n=2):
    inst=Instances((32,32))
    inst.pred_boxes=Boxes(torch.zeros(n,4))
    inst.reid_features=torch.arange(n*1152,dtype=torch.float32).view(n,1152)
    inst.scores=torch.ones(n)
    inst.pred_classes=torch.zeros(n,dtype=torch.long)
    if ids is not None: inst.track_ids=torch.tensor(ids,dtype=torch.long)
    return inst


def model(keep=False):
    result=SimpleNamespace(overlap_thresh=.1,not_mult_thresh=False,with_bank=True,bank_size=2,
        thred_bank=.4,roi_heads=SimpleNamespace(_activate_asso=lambda a:a),
        cir_controller=IdentityRevisionController(SimpleNamespace(ENABLED=keep,MODE='keep' if keep else 'off',
            ENABLE_PROVISIONAL=False,ENABLE_REVISION=False,ENABLE_RELINK=False,REVISION_WINDOW=0)))
    for name in ('compute_native_track_scores','solve_native_assignment','propose_identity_revision',
                 'resolve_bank_fallback','commit_native_identity_state','run_memory_tracker','memory_bank'):
        setattr(result,name,MethodType(getattr(native.GTRRCNN,name),result))
    return result


class NativeCommitTests(unittest.TestCase):
    def compare(self, score, keep=False, bank=False):
        results=[]
        for method in (original_method(),native.GTRRCNN.run_global_tracker_plus):
            native.poss_ids.poss_ids=set();native.old_ids.old_ids=set();native.old_reids.old_reids=[]
            m=model(keep);q,h=score.shape
            history=instance(list(range(1,h+1)),h)
            current=instance(n=q);instances=[history,current]
            counts={i:1 for i in range(1,h+1)};reids={i:history[i-1] for i in counts}
            if bank:
                counts[40]=2;reids[40]=instance([40,40]);native.poss_ids.poss_ids.add(40)
                # Actual memory_bank + run_memory_tracker; only transformer scores stubbed.
                m.get_asso=lambda inst,k,geo_context=False: ([torch.full((len(inst[-1]),len(inst[0])),.8)],None,None,None,None)
            output=method(m,2,instances,[score.clone()],None,1,max(counts,default=0),counts,reids,
                          [instance([1],1)] if bank else [],0)
            results.append(fingerprint(dict(instances=output[0],id_count=output[1],counts=counts,
                  reids=reids,poss=native.poss_ids.poss_ids,old=native.old_reids.old_reids)))
        self.assertEqual(*results)

    def test_off_nonempty(self): self.compare(torch.tensor([[.8,.2],[.3,.9]]))
    def test_keep_nonempty(self): self.compare(torch.tensor([[.8,.2],[.3,.9]]),keep=True)
    def test_strict_equal_threshold(self): self.compare(torch.tensor([[.1]]))
    def test_joint_competition(self): self.compare(torch.tensor([[.9,.7],[.85,.01]]))
    def test_new_births(self): self.compare(torch.zeros(3,2))
    def test_empty_detection(self): self.compare(torch.empty(0,2))
    def test_empty_candidates(self): self.compare(torch.empty(2,0))
    def test_empty_all(self): self.compare(torch.empty(0,0))
    def test_real_bank_recovery(self): self.compare(torch.zeros(2,2),bank=True)
    def test_keep_real_bank(self): self.compare(torch.zeros(2,2),keep=True,bank=True)
    def test_checkpoint_contract(self):
        from detectron2.config import get_cfg
        from gtr.config import add_gtr_config
        cfg=get_cfg();add_gtr_config(cfg)
        self.assertFalse(cfg.MODEL.CIR.ENABLED)
        self.assertFalse(cfg.MODEL.CIR.ENABLE_REVISION)
    def test_keep_identity_object(self):
        x=torch.zeros(2,3);m=model(True)
        self.assertIs(m.propose_identity_revision(x),x)

    def test_c1_native_joint_resolve_and_dictionary_commit(self):
        from unittest.mock import patch
        m=model(True);m.cir_controller.mode='c1'
        history=instance([1,2]);current=instance()
        record=dict(video_id=4,frame_id=1,view_id=1,image_id=1)
        m.cir_controller.observe([history],[record])
        m.cir_controller.observe([current],[dict(record,frame_id=2,image_id=2)])
        counts={1:1,2:1};reids={1:history[0],2:history[1]}
        native.poss_ids.poss_ids=set();native.old_reids.old_reids=[]
        real=native.linear_sum_assignment
        with patch.object(native,'linear_sum_assignment',wraps=real) as solver:
            output=native.GTRRCNN.run_global_tracker_plus(m,1,[history,current],
                [torch.zeros(2,2)],None,1,2,counts,reids,[],0)
        self.assertEqual(solver.call_count,2)
        self.assertEqual(set(output[0][-1].track_ids.tolist()),{1,2})
        self.assertEqual(output[1],2);self.assertEqual(counts,{1:2,2:2})
        self.assertEqual([len(reids[i]) for i in (1,2)],[2,2])
        self.assertEqual(m.cir_controller.extra_joint_solves,1)

    def test_missing_provenance_keeps_original_joint_result(self):
        m=model(True);m.cir_controller.mode='c1';m.cir_controller.timestamp=2;m.cir_controller.video=4
        h=instance([1],1);q=instance(n=1);counts={1:1};reids={1:h[0]}
        native.poss_ids.poss_ids=set();native.old_reids.old_reids=[]
        output=native.GTRRCNN.run_global_tracker_plus(m,1,[h,q],[torch.tensor([[.9]])],None,1,1,counts,reids,[],0)
        self.assertEqual(output[0][-1].track_ids.tolist(),[1])
        self.assertEqual(m.cir_controller.extra_joint_solves,0)
        self.assertEqual(sum(m.cir_controller.fallbacks.values()),1)

    def test_bias2_reuses_native_solve_and_commit_without_identity_features(self):
        m=model(True);m.cir_controller.mode='bias2'
        h=instance([1,2]);q=instance();counts={1:1,2:1};reids={1:h[0],2:h[1]}
        native.poss_ids.poss_ids=set();native.old_reids.old_reids=[]
        output=native.GTRRCNN.run_global_tracker_plus(m,1,[h,q],[torch.zeros(2,2)],None,1,2,counts,reids,[],0)
        self.assertEqual(set(output[0][-1].track_ids.tolist()),{1,2})
        self.assertEqual(counts,{1:2,2:2});self.assertEqual(output[1],2)
        self.assertEqual(m.cir_controller.extra_joint_solves,1)
        self.assertIsNone(m.cir_controller.timestamp)


if __name__=='__main__':unittest.main()
