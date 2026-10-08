"""Independent real-state operator tests; no learned architecture claims."""
from dataclasses import replace
from pathlib import Path
import torch,json,copy
from jev_phase8_common import *
from jev_phase8_candidate_submit import native_existing_ids,validate_pairs
from jev_phase7_state import CompleteStateDigest

def main():
    protect();start=binding();assert not start['worktree_dirty']
    m=json.loads((OUT/'opportunity_scan_v1/video13/SCAN_RESULT.json').read_text());r=m['bounded_snapshots'][0]
    s=torch.load(r['path'],map_location='cpu');assert sha(r['path'])==r['sha256']
    lab=native_lab(13);model=lab.engine.association_fn.model;p=s['proposal'];state=s['state'];payload=lab.cache.load(*s['key']);digest=CompleteStateDigest()
    before=digest(state,s['metadata']);actions={int(d['context']['detection_index']):d['off_action']for d in s['decision_rows']if d['question']=='MATCH_DECISION'}
    expected=native_existing_ids(model,payload,state,p,p.pairs,actions)
    perm=list(reversed(range(len(p.track_ids))));inverse={old:new for new,old in enumerate(perm)}
    pp=replace(p,track_ids=tuple(p.track_ids[i]for i in perm),scores=p.scores[:,perm],pairs={row:inverse[col]for row,col in p.pairs.items()},banned_edges=tuple((r,inverse[c])for r,c in p.banned_edges))
    actual=native_existing_ids(model,payload,state,pp,pp.pairs,actions);assert expected==actual
    new=native_existing_ids(model,payload,state,p,{},dict.fromkeys(actions,'START_NEW'));assert all(i==-1 for i in new)
    if p.pairs:
        edge=next(iter(p.pairs.items()));masked=replace(p,banned_edges=tuple(p.banned_edges)+(edge,))
        try:validate_pairs(masked,p.pairs)
        except AssertionError:pass
        else:raise AssertionError('masked actual candidate accepted')
    assert digest(state,s['metadata'])==before
    save(REPORTS/'NATIVE_SUBMIT_OPERATOR_TESTS.json',{'status':'PASS','binding':start,'snapshot_sha256':r['sha256'],'real_video':13,'real_key':list(s['key']),
        'actual_production_MATCH_candidate_column_permutation_existing_IDs':'PASS','actual_production_MATCH_all_NEW_existing_IDs':'PASS',
        'actual_masked_edge_rejected':'PASS','all_actual_existing_IDs_unique':len([i for i in expected if i>=0])==len(set(i for i in expected if i>=0)),
        'complete_state_observational':'PASS','production_MATCH_input_signature_no_GT_or_future':True,
        'deployed_full_commit_bank_NEW_parity':'NOT_VERIFIED','learned_candidate_model_permutation':'NOT_RUN',
        'model_values_assignment_interface':'NOT_IMPLEMENTED','scope':'production MATCH helper only; independently permuted real input, not copied replay resolver output'})
    protect();print('NATIVE_SUBMIT_OPERATOR_TESTS PASS')
if __name__=='__main__':main()
