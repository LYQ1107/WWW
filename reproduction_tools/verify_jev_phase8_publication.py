"""Independent artifact, admission, protected-source and publication guards."""
import json,subprocess,os
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty'],'commit reviewable source/results before verification'
    required_docs=['FINAL_GOAL','RESEARCH_PLAN','CODE_AUDIT','CORRECTIVE_OPPORTUNITY_REPORT','DATASET_CONTRACT','FAILURE_CAUSE_ANALYSIS','CANDIDATE_JEV_ARCHITECTURE','FAIR_BASELINE_PROTOCOL','FINAL_RESEARCH_REPORT']
    required_reports=['STORAGE_BEFORE','STORAGE_AFTER','SAFE_DELETE_MANIFEST','CORRECTIVE_OPPORTUNITY_AUDIT','NATIVE_CORRECTIVE_V2_MANIFEST','NATIVE_CORRECTIVE_V2_AUDIT','LEARNING_CURVES','DATA_SCALING','CAPACITY_ABLATION','SAMPLING_LOSS_ABLATION','ARCHITECTURE_ABLATION','NATIVE_PARITY','HELDOUT_RESULTS','GO_NO_GO']
    assert all((ROOT/'docs'/f'PHASE8_{n}.md').is_file()for n in required_docs)
    assert all((REPORTS/f'{n}.json').is_file()for n in required_reports)
    assert all((REPORTS/n).is_file()for n in('CHECKPOINT_INVENTORY.csv','DELETE_DRY_RUN.md','STORAGE_CLEANUP_REPORT.md'))
    audit=json.loads((REPORTS/'NATIVE_CORRECTIVE_ORACLE_AUDIT.json').read_text());scan=json.loads((REPORTS/'CORRECTIVE_OPPORTUNITY_AUDIT.json').read_text())
    assert len(audit['events'])==95
    assert len({tuple(e['key'])+(e['row'],)for e in audit['events']})==95
    for e in audit['events']:
        for tag in e['verified_branches']:
            b=e['branches'][tag];assert b['desired_candidate_committed']and b['immediate_anchored_correct']and b['H32_complete']
            assert b['native_MATCH_existing_ID_before_bank_recovery']==b['committed_ID_metadata']
            assert b['committed_ID_metadata']in e['correct_candidate_ID_metadata']
            assert b['horizons']['32']['delta_utility']>0 and b['horizons']['32']['delta_utility_birth_zero']>0
            assert b['all_camera_payload_IDs_unique']
    protected=json.loads((REPORTS/'CODE_CONTRACT_AUDIT.json').read_text())['frozen_source_sha256']
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    inputs={'foundation':('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth',json.loads(PREREG.read_text())['foundation_sha256']),
       'cache_index':('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train/index.jsonl',json.loads(PREREG.read_text())['cache_index_sha256']),
       'official_TRAIN_annotation':('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json','9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85')}
    assert all(sha(p)==h for p,h in inputs.values())
    artifacts=scan['raw_manifest']+audit['raw_manifest'];seen=set()
    for a in artifacts:
        if a['path']in seen:continue
        seen.add(a['path']);assert sha(a['path'])==a['sha256'];assert Path(a['path']).stat().st_size==a['bytes']
    for n in ['LEARNING_CURVES','DATA_SCALING','CAPACITY_ABLATION','SAMPLING_LOSS_ABLATION','NORMALIZATION_ABLATION','ARCHITECTURE_ABLATION','HELDOUT_RESULTS']:
        x=json.loads((REPORTS/f'{n}.json').read_text());assert x['status']=='NOT_RUN'and x['metrics']is None and x['checkpoints_created']==0
    assert not audit['formal_data_gate']['pass'];assert audit['validation']['verified_events']==13
    files=subprocess.check_output(['git','diff','--name-only', 'ecc0e63','HEAD'],cwd=ROOT,text=True).splitlines()
    assert not any(p.endswith(('.pth','.pt','.mp4','.npz','.gz'))or'/evidence/'in p or'/raw/'in p for p in files)
    assert not any(p.startswith(('gtr/','configs/','reports/JEV_PHASE5/','reports/JEV_PHASE6/','reports/JEV_PHASE7/'))for p in files)
    result={'status':'PASS','binding':start,'verified_events':len(audit['events']),'verified_raw_artifacts':len(seen),'protected_source_checks':len(protected),'frozen_input_sha_checks':inputs,
        'required_files_complete':True,'Gate_fail_and_NOT_RUN_honest':True,'no_new_large_binary_publication':True,'phase5_6_7_and_B2_protected':True,
        'matched_submission_origin_and_positive_H32_birth_zero':True,'source_and_result_paths_hashes_verified':True,'changed_review_files':len(files),'changed_review_bytes':sum((ROOT/p).stat().st_size for p in files if(ROOT/p).is_file())}
    save(REPORTS/'PUBLICATION_VERIFICATION.json',result);protect();print(json.dumps({k:result[k]for k in('status','verified_events','verified_raw_artifacts','changed_review_bytes')}))
if __name__=='__main__':main()
