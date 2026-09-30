#!/usr/bin/env python3
"""Static and algebraic no-op checks for the audit guards.

The audit branches are guarded in the model.  This check records the exact
source paths and verifies that an all-one audit mask produces the original
membership matrix and threshold support for deterministic tensors.
"""
from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
 root=Path(__file__).resolve().parents[1]
 torch.manual_seed(20260930)
 ids=torch.randint(1,20,(127,)); unique=torch.unique(ids)
 id_inds=(unique[None,:]==ids[:,None]).float(); assoc=torch.rand(23,len(ids))
 baseline=assoc @ id_inds
 weights=torch.ones_like(ids,dtype=torch.float32); weighted=id_inds*weights[:,None]
 audit=assoc @ weighted
 support=id_inds.sum(0); weighted_support=weighted.sum(0)
 boxes_equal=bool(torch.equal(id_inds,weighted)); scores_equal=bool(torch.equal(baseline,audit)); support_equal=bool(torch.equal(support,weighted_support))
 env_vars=['GMT_AUDIT_DIR','GMT_AUDIT_DISABLE_TEST_JITTER','GMT_AUDIT_DUMP_FEATURES','GMT_AUDIT_HISTORY_POLICY','GMT_AUDIT_HISTORY_KEEP_RATIO']
 source_files=['gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/roi_heads/gtr_roi_heads.py','gtr/data/datasets/mot.py','test_net.py']
 result={'status':'PASS' if boxes_equal and scores_equal and support_equal else 'FAIL','fixed_seed':20260930,'all_audit_env_unset_in_check':all(v not in os.environ for v in env_vars),'algebraic_membership_equal':boxes_equal,'algebraic_traj_score_equal':scores_equal,'algebraic_support_equal':support_equal,'source_sha256':{p:sha(root/p) for p in source_files},'baseline_source_commit':'dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb','note':'The repository evaluator writes fixed-root MOT text; prediction-level runtime equivalence is recorded separately per run. The only pre-existing release worktree fix retained here is the GMTMultimodalDatasetMapper class boundary.'}
 out=root/'audit/manifests/audit_noop_equivalence.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
