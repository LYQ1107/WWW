"""Record actual frozen production/adapter contracts without claiming new parity."""
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports/JEV_PHASE8'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    anchors={'jev/www-jev-phase5-20261007':'a37083dac0a23eb3846cb252eeb975982aaaabc9',
       'jev/www-jev-lifecycle-phase6-20261008':'40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e',
       'jev/www-jev-phase7-causal-structured-20261008':'ecc0e63f544cbbc272797c17ea285b0b8cce5f80'}
    for branch,expected in anchors.items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==expected
    b2=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
    assert sha(b2)=='f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
    paths=['gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/jev_assignment.py','gtr/modeling/jev_state.py',
           'reproduction_tools/jev_counterfactual_v2.py','reproduction_tools/run_early_pilot_tracking.py',
           'reproduction_tools/jev_phase6_rollouts.py','reproduction_tools/jev_phase6_native_match.py',
           'reproduction_tools/jev_phase7_native.py','reproduction_tools/jev_phase7_state.py','reproduction_tools/jev_phase7_offline.py']
    evidence={'status':'CODE_CONTRACT_AUDIT_COMPLETE_NEW_INTERFACE_NOT_YET_VERIFIED','protected_heads':anchors,
       'B2_sha256':sha(b2),'frozen_source_sha256':{p:sha(ROOT/p)for p in paths},'preregistration_sha256':sha(REPORTS/'PREREGISTRATION.json'),
       'formal_association':'actual GMT foundation/transformer with branch-local trajectory RNG and fixed TRAIN perception cache',
       'original_assignment':'gtr.modeling.jev_assignment.constrained_hungarian, raw GMT score maximization with rejected-edge mask',
       'production_match_hook':'GTRRCNN._apply_jev_match_decisions; initial A/R/NEW triage and binary second validation',
       'mutable_commit':'jev_counterfactual_v2 engine resolve_actions/step; supplied proposal and RNG provenance reused, then memory/birth/stale/bank transitions',
       'critical_limitation':'the mutable research resolver is not itself proof of production candidate-choice parity; new direct candidate values require a tested shared production operator and native commit/state parity',
       'cross_view_uniqueness':'one identity per detection set in a single frame/view; lawful identity sharing between cameras remains allowed',
       'candidate_semantics':'track IDs are metadata; current available GMT track_ids and exact raw scores only; no synthetic stale/correct identity appended',
       'coordinate_contract':'cache boxes retain original image_size, frame/view cache0 versus annotation1; offline per-image IoU>=0.5 one-to-one GT alignment',
       'offline_semantic_mapping':'historical anchored GT from preceding committed detections; complete-purity>=2; multiple predicted aliases retained; equality of ID integers never used',
       'factual_phase8_native_parity':'NOT_RUN','direct_candidate_native_parity':'NOT_RUN','GT_runtime_actor_input':False,
       'production_gtr_modified_at_audit':bool(subprocess.check_output(['git','diff','ecc0e63f544cbbc272797c17ea285b0b8cce5f80','--name-only','--','gtr'],cwd=ROOT,text=True).strip()),
       'Full24_started':False,'official_TEST_read':False,'existing_phase7_negative_conclusion_preserved':True}
    (REPORTS/'CODE_CONTRACT_AUDIT.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (ROOT/'docs/PHASE8_CODE_AUDIT.md').write_text('''# Phase VIII code audit

Phase V/VI/VII heads and permanent B2 match the required anchors locally and on GitHub. The exact source inventory and frozen protocol hashes are in CODE_CONTRACT_AUDIT.json; original production files are unchanged at this audit.

The formal backend executes the actual GMT association transformer on immutable cached perception, preserving branch-local trajectory RNG. `constrained_hungarian` maximizes raw scores and masks rejected edges. The production `GTRRCNN._apply_jev_match_decisions` currently supplies A/R/NEW plus binary second validation; it does not establish an arbitrary candidate-value submit interface. `jev_counterfactual_v2` proposes/resolves and commits real mutable history, births, memory and bank transitions, but copying its research resolution alone is insufficient production parity evidence.

Phase VIII must measure candidate availability, global feasibility and actual native correction separately. Candidate IDs are indexing metadata, not GT identities or neural scalar features. One-to-one uniqueness applies per frame/view, while consistent cross-camera identity reuse is legal. No absent detection or stale candidate may be fabricated. Every intervention must reuse current raw GMT tensor/candidate order and prefix RNG; subsequent scores naturally depend on its mutated state.

Offline GT alignment follows the original image size, cache frame/view zero versus annotation one and per-image IoU>=0.5. Historical semantic anchors use only preceding committed known observations, require at least two with complete purity and agreement of first/majority identity, retain alias ambiguity, and mark mixed/unanchored histories UNKNOWN. These annotation labels are diagnostic/supervision fields separated from the live actor. New factual/state, direct-submit, masking/NEW/collision/RNG/no-leakage checks are NOT_RUN until executed; no new production-parity claim is made here.

Phase VII's rule-superior, ACCEPT-only and unidentifiable memory/react findings remain negative. Candidate JEV and Unified have never been trained. The original H8 shortcut labels do not become native v2 truth. Full24 and official TEST are not authorized.
''')
    print(json.dumps({'status':evidence['status'],'protected_anchors':'PASS','frozen_source_files':len(paths)}))
if __name__=='__main__':main()
